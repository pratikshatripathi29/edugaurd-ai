"""
EduGuard AI — Explainability & Feature Attribution Module (Stage 5)

This module provides game-theoretic feature attribution using TreeSHAP
(SHapley Additive exPlanations) for both classification (at-risk detection)
and regression (final grade G3 prediction).

Key Architecture & Leakage Safeguards:
1. Pipeline Awareness:
   Models in EduGuard AI are encapsulated in scikit-learn Pipelines with
   ColumnTransformers (and wrapped in CalibratedClassifierCV for classification).
   TreeSHAP requires the raw tree model and preprocessed feature matrix.
   This module unwraps the fitted preprocessor, transforms raw feature rows,
   restores human-readable feature names, and computes SHAP values on the
   transformed feature space.
2. Output Scale Transparency:
   - RandomForestClassifier: Values represent additive contributions to predicted
     class probability (P(at_risk)), bounded in [0, 1].
   - XGBClassifier: Values represent additive contributions to margin / log-odds
     (logit = ln(p / (1 - p))).
   - RandomForestRegressor: Values represent additive contributions to final grade
     points (G3 in [0, 20]).
3. Explainers are cached in-memory so Streamlit and interactive calls remain snappy.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path when invoked directly
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

from src.config import FIGURES_DIR, LEAKAGE_COLS, RANDOM_STATE
from src.features import prepare_scenario_data

# In-memory explainer cache for high-performance interactive & Streamlit use
_EXPLAINER_CACHE: Dict[Tuple[str, str], Tuple[shap.TreeExplainer, Any, List[str]]] = {}


def clear_explainer_cache() -> None:
    """Clear all cached TreeExplainers."""
    global _EXPLAINER_CACHE
    _EXPLAINER_CACHE.clear()


def unwrap_pipeline_and_tree(
    model_or_pipeline: Any,
) -> Tuple[Any, Any, bool]:
    """Extract fitted preprocessor and tree estimator from wrapped pipelines.

    Handles:
    - CalibratedClassifierCV wrapping FrozenEstimator or Pipeline
    - Standard Pipeline with ('preprocessor', preprocessor) and ('clf' or 'regressor')
    - Standalone tree estimators

    Args:
        model_or_pipeline: Fitted classifier, regressor, Pipeline, or CalibratedClassifierCV.

    Returns:
        Tuple of (preprocessor, tree_estimator, is_classifier).

    Raises:
        ValueError: If preprocessor or tree estimator cannot be identified.
    """
    is_classifier = True

    # Case 1: CalibratedClassifierCV
    if isinstance(model_or_pipeline, CalibratedClassifierCV):
        if not hasattr(model_or_pipeline, "calibrated_classifiers_") or len(model_or_pipeline.calibrated_classifiers_) == 0:
            raise ValueError("CalibratedClassifierCV estimator is not fitted.")
        calibrated_item = model_or_pipeline.calibrated_classifiers_[0]
        base_pipe = getattr(calibrated_item, "estimator", None)
        # Unwrap FrozenEstimator if present (scikit-learn 1.4+)
        if hasattr(base_pipe, "estimator"):
            base_pipe = base_pipe.estimator
    else:
        base_pipe = model_or_pipeline

    # Case 2: Pipeline
    if isinstance(base_pipe, Pipeline):
        if "preprocessor" not in base_pipe.named_steps:
            raise ValueError("Pipeline does not contain a 'preprocessor' step.")
        preprocessor = base_pipe.named_steps["preprocessor"]

        if "clf" in base_pipe.named_steps:
            tree_estimator = base_pipe.named_steps["clf"]
            is_classifier = True
        elif "regressor" in base_pipe.named_steps:
            tree_estimator = base_pipe.named_steps["regressor"]
            is_classifier = False
        else:
            # Fallback to last step
            step_name, tree_estimator = base_pipe.steps[-1]
            is_classifier = "clf" in step_name.lower() or hasattr(tree_estimator, "predict_proba")
        return preprocessor, tree_estimator, is_classifier

    # Case 3: Bare estimator
    is_classifier = hasattr(base_pipe, "predict_proba")
    return None, base_pipe, is_classifier


def get_output_scale_info(tree_estimator: Any, is_classifier: bool = True) -> Dict[str, str]:
    """Explain which mathematical output scale SHAP values represent.

    Args:
        tree_estimator: The underlying tree model (RandomForest, XGBoost, etc.).
        is_classifier: Whether the model is a classifier.

    Returns:
        Dictionary detailing scale name, unit, and educational interpretation.
    """
    est_name = type(tree_estimator).__name__

    if not is_classifier:
        return {
            "scale": "grade_points",
            "unit": "final grade points (0 - 20 scale)",
            "estimator": est_name,
            "interpretation": (
                "SHAP values measure the exact additive change in predicted final grade points (G3). "
                "A value of +1.5 means this feature added 1.5 points to the predicted score."
            ),
        }

    if "XGB" in est_name:
        return {
            "scale": "log_odds",
            "unit": "margin / log-odds (logit = ln(p / (1 - p)))",
            "estimator": est_name,
            "interpretation": (
                "For XGBoost, TreeExplainer computes values on the raw tree margin (log-odds). "
                "Positive values increase the log-odds of being at-risk; negative values decrease it."
            ),
        }

    # Default for RandomForestClassifier
    return {
        "scale": "probability",
        "unit": "predicted probability of academic risk [0.0, 1.0]",
        "estimator": est_name,
        "interpretation": (
            "For RandomForestClassifier, TreeExplainer outputs values in probability space. "
            "A SHAP value of +0.12 means this feature increased the predicted probability of failure by 12 percentage points."
        ),
    }


def get_transformed_features(
    preprocessor: Any,
    df: pd.DataFrame,
    auto_drop_leakage: bool = False,
) -> Tuple[np.ndarray, List[str]]:
    """Transform input features using fitted preprocessor and extract readable names.

    Args:
        preprocessor: Fitted ColumnTransformer from training split.
        df: Input DataFrame containing student feature columns.
        auto_drop_leakage: If True, automatically strips leakage columns; if False, raises ValueError.

    Returns:
        Tuple of (X_transformed as ndarray, feature_names as list of str).

    Raises:
        ValueError: If input DataFrame contains target leakage columns and auto_drop_leakage is False.
    """
    clean_df = df.copy()

    # Guard against target leakage
    leakage_present = [leak for leak in LEAKAGE_COLS if leak in clean_df.columns]
    if leakage_present:
        if auto_drop_leakage:
            clean_df = clean_df.drop(columns=leakage_present, errors="ignore")
        else:
            raise ValueError(
                f"TARGET LEAKAGE DETECTED in explainability pipeline: {leakage_present} found in input data."
            )

    # Clean student key or other administrative columns if present
    drop_candidates = ["student_key", "is_simulated", "student_id", "data_source"]
    clean_df = clean_df.drop(columns=[c for c in drop_candidates if c in clean_df.columns], errors="ignore")

    # Automatically compute engineered features if missing from input
    from src.features import add_engineered_features
    engineered_cols = {"alcohol_total", "study_time_ratio", "support_total", "failures_flag", "high_absence"}
    has_trend = "grade_trend" in [c for c in getattr(preprocessor, "feature_names_in_", [])]
    if not engineered_cols.issubset(set(clean_df.columns)) or (has_trend and "grade_trend" not in clean_df.columns):
        clean_df = add_engineered_features(clean_df, include_trend=has_trend)

    X_trans = preprocessor.transform(clean_df)
    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()

    if hasattr(preprocessor, "get_feature_names_out"):
        feature_names = list(preprocessor.get_feature_names_out())
    else:
        feature_names = [f"feature_{i}" for i in range(X_trans.shape[1])]

    return X_trans, feature_names


def get_cached_explainer(
    scenario: str = "A",
    model_type: str = "classification",
    model: Optional[Any] = None,
) -> Tuple[shap.TreeExplainer, Any, List[str]]:
    """Retrieve or build a cached TreeExplainer and preprocessor.

    Args:
        scenario: 'A' (Early Warning) or 'B' (Mid-Semester).
        model_type: 'classification' or 'regression'.
        model: Optional pre-loaded model. If None, loads from models/.

    Returns:
        Tuple of (TreeExplainer, fitted_preprocessor, feature_names).
    """
    key = (model_type.lower(), scenario.upper())
    if key in _EXPLAINER_CACHE:
        return _EXPLAINER_CACHE[key]

    # Load model if not provided
    if model is None:
        if model_type.lower() == "classification":
            from src.classification import load_classification_model
            model, _ = load_classification_model(scenario)
        else:
            from src.regression import load_model
            model, _ = load_model(scenario)

    preprocessor, tree_estimator, _ = unwrap_pipeline_and_tree(model)

    explainer = shap.TreeExplainer(tree_estimator)
    feature_names = list(preprocessor.get_feature_names_out())

    _EXPLAINER_CACHE[key] = (explainer, preprocessor, feature_names)
    return explainer, preprocessor, feature_names


def compute_shap_explanation(
    explainer: shap.TreeExplainer,
    X_trans: np.ndarray,
    feature_names: Optional[List[str]] = None,
    is_classifier: bool = True,
) -> shap.Explanation:
    """Compute SHAP Explanation object focused on the target outcome.

    For binary classification, extracts class 1 (at-risk) attributions.
    For regression, extracts continuous grade point attributions.

    Args:
        explainer: Fitted shap.TreeExplainer.
        X_trans: Transformed feature array of shape (N, D).
        feature_names: Optional readable feature names.
        is_classifier: Whether model is a classifier.

    Returns:
        shap.Explanation object of shape (N, D).
    """
    raw_exp = explainer(X_trans)

    # Binary classifier with shape (N, D, 2)
    if is_classifier and len(raw_exp.shape) == 3 and raw_exp.shape[2] == 2:
        exp = raw_exp[..., 1]
    else:
        exp = raw_exp

    if feature_names is not None:
        exp.feature_names = list(feature_names)

    return exp


def global_importance(
    explainer: shap.TreeExplainer,
    X_trans: np.ndarray,
    feature_names: Optional[List[str]] = None,
    is_classifier: bool = True,
) -> pd.DataFrame:
    """Compute mean absolute SHAP values per feature across all instances.

    Args:
        explainer: Fitted shap.TreeExplainer.
        X_trans: Transformed feature array.
        feature_names: Feature names list.
        is_classifier: Whether model is a classifier.

    Returns:
        pd.DataFrame with ['feature', 'mean_abs_shap'] sorted descending.
    """
    exp = compute_shap_explanation(explainer, X_trans, feature_names, is_classifier=is_classifier)
    names = feature_names if feature_names is not None else [f"feature_{i}" for i in range(X_trans.shape[1])]

    mean_abs = np.mean(np.abs(exp.values), axis=0)
    df_imp = pd.DataFrame({
        "feature": names,
        "mean_abs_shap": mean_abs,
    }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)

    return df_imp


def summary_plot(
    explainer: shap.TreeExplainer,
    X_trans: np.ndarray,
    feature_names: Optional[List[str]] = None,
    is_classifier: bool = True,
    max_display: int = 15,
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> plt.Figure:
    """Generate and optionally save a SHAP beeswarm summary plot.

    Args:
        explainer: Fitted TreeExplainer.
        X_trans: Transformed feature array.
        feature_names: Readable feature names.
        is_classifier: Whether model is classifier.
        max_display: Max features to display.
        title: Optional plot title.
        save_path: File path to save figure.
        show: Whether to call plt.show().

    Returns:
        matplotlib Figure object.
    """
    exp = compute_shap_explanation(explainer, X_trans, feature_names, is_classifier=is_classifier)

    fig = plt.figure(figsize=(10, 7))
    shap.plots.beeswarm(exp, max_display=max_display, show=False)
    if title:
        plt.title(title, fontsize=12, fontweight="bold", pad=15)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def bar_plot(
    explainer: shap.TreeExplainer,
    X_trans: np.ndarray,
    feature_names: Optional[List[str]] = None,
    is_classifier: bool = True,
    max_display: int = 15,
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> plt.Figure:
    """Generate and optionally save a SHAP mean absolute bar plot.

    Args:
        explainer: Fitted TreeExplainer.
        X_trans: Transformed feature array.
        feature_names: Readable feature names.
        is_classifier: Whether model is classifier.
        max_display: Max features to display.
        title: Optional plot title.
        save_path: File path to save figure.
        show: Whether to call plt.show().

    Returns:
        matplotlib Figure object.
    """
    exp = compute_shap_explanation(explainer, X_trans, feature_names, is_classifier=is_classifier)

    fig = plt.figure(figsize=(10, 6))
    shap.plots.bar(exp, max_display=max_display, show=False)
    if title:
        plt.title(title, fontsize=12, fontweight="bold", pad=15)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def waterfall(
    explainer: shap.TreeExplainer,
    X_trans: np.ndarray,
    row_idx: int = 0,
    feature_names: Optional[List[str]] = None,
    is_classifier: bool = True,
    max_display: int = 10,
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> plt.Figure:
    """Generate and optionally save a local SHAP waterfall plot for a single student.

    Args:
        explainer: Fitted TreeExplainer.
        X_trans: Transformed feature array.
        row_idx: Index of student row to explain.
        feature_names: Readable feature names.
        is_classifier: Whether model is classifier.
        max_display: Max features to display.
        title: Optional plot title.
        save_path: File path to save figure.
        show: Whether to call plt.show().

    Returns:
        matplotlib Figure object.
    """
    exp = compute_shap_explanation(explainer, X_trans, feature_names, is_classifier=is_classifier)
    single_exp = exp[row_idx]

    fig = plt.figure(figsize=(10, 6))
    shap.plots.waterfall(single_exp, max_display=max_display, show=False)
    if title:
        plt.title(title, fontsize=12, fontweight="bold", pad=15)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def explain_student(
    row: Union[pd.Series, pd.DataFrame],
    explainer: shap.TreeExplainer,
    preprocessor: Any,
    feature_names: Optional[List[str]] = None,
    top_n: int = 5,
    is_classifier: bool = True,
) -> Dict[str, Any]:
    """Generate a detailed, honest local explanation for a single student profile.

    Computes exact real SHAP values and outputs:
    1. A structured contributions DataFrame.
    2. Top positive risk factors.
    3. Top negative (protective) risk factors.
    4. A factual, plain-English summary derived entirely from real computed numbers.

    Args:
        row: Single student profile as Series or 1-row DataFrame.
        explainer: Fitted TreeExplainer.
        preprocessor: Fitted preprocessor.
        feature_names: Optional feature names.
        top_n: Number of top features to report in positive/negative lists.
        is_classifier: True for at-risk classification; False for G3 regression.

    Returns:
        Dictionary containing real attribution values and plain-language summary.
    """
    # Convert Series to single-row DataFrame if needed
    if isinstance(row, pd.Series):
        df_single = pd.DataFrame([row])
    else:
        df_single = row.iloc[[0]].copy()

    # Preprocess and transform with auto-dropping of target columns if present
    X_trans, names = get_transformed_features(preprocessor, df_single, auto_drop_leakage=True)
    if feature_names is None:
        feature_names = names

    # Compute explanation
    exp = compute_shap_explanation(explainer, X_trans, feature_names, is_classifier=is_classifier)
    single_exp = exp[0]

    shap_vals = single_exp.values
    base_val = float(np.squeeze(single_exp.base_values))
    predicted_val = float(base_val + np.sum(shap_vals))
    total_shift = float(predicted_val - base_val)

    # Build contributions DataFrame
    df_contrib = pd.DataFrame({
        "feature": feature_names,
        "transformed_value": X_trans[0],
        "shap_value": shap_vals,
        "abs_impact": np.abs(shap_vals),
    })

    # Classify direction
    if is_classifier:
        df_contrib["direction"] = np.where(
            df_contrib["shap_value"] > 0, "Elevates Risk", "Mitigates Risk"
        )
    else:
        df_contrib["direction"] = np.where(
            df_contrib["shap_value"] > 0, "Increases Score", "Decreases Score"
        )

    # Sort
    top_pos = (
        df_contrib[df_contrib["shap_value"] > 0]
        .sort_values(by="shap_value", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    top_neg = (
        df_contrib[df_contrib["shap_value"] < 0]
        .sort_values(by="shap_value", ascending=True)
        .head(top_n)
        .reset_index(drop=True)
    )

    # Generate fact-checked plain-English summary
    summary_sentences = []
    if is_classifier:
        summary_sentences.append(
            f"Cohort baseline risk is {base_val * 100:.1f}%. "
            f"The model estimates this student's risk at {predicted_val * 100:.1f}% "
            f"({'+' if total_shift >= 0 else ''}{total_shift * 100:.1f}% net shift)."
        )

        if not top_pos.empty:
            pos_phrases = [
                f"{row_item['feature']} (+{row_item['shap_value'] * 100:.1f}% risk)"
                for _, row_item in top_pos.iterrows()
            ]
            summary_sentences.append(
                f"Primary factors elevating academic risk: {', '.join(pos_phrases)}."
            )
        else:
            summary_sentences.append("No strong features elevating academic risk were identified.")

        if not top_neg.empty:
            neg_phrases = [
                f"{row_item['feature']} ({row_item['shap_value'] * 100:.1f}% risk)"
                for _, row_item in top_neg.iterrows()
            ]
            summary_sentences.append(
                f"Protective factors mitigating academic risk: {', '.join(neg_phrases)}."
            )
        else:
            summary_sentences.append("No strong protective factors were identified.")

        summary_sentences.append(
            "Advisory Note: These SHAP values indicate model attributions, not direct causation. "
            "They should guide counseling conversations, not punitive measures."
        )
    else:
        summary_sentences.append(
            f"Cohort baseline final score is {base_val:.2f} points. "
            f"The model estimates this student's final score at {predicted_val:.2f} / 20 "
            f"({'+' if total_shift >= 0 else ''}{total_shift:.2f} points shift)."
        )

        if not top_pos.empty:
            pos_phrases = [
                f"{row_item['feature']} (+{row_item['shap_value']:.2f} pts)"
                for _, row_item in top_pos.iterrows()
            ]
            summary_sentences.append(
                f"Features boosting predicted grade: {', '.join(pos_phrases)}."
            )

        if not top_neg.empty:
            neg_phrases = [
                f"{row_item['feature']} ({row_item['shap_value']:.2f} pts)"
                for _, row_item in top_neg.iterrows()
            ]
            summary_sentences.append(
                f"Features pulling grade downward: {', '.join(neg_phrases)}."
            )

    plain_english = " ".join(summary_sentences)

    return {
        "base_value": base_val,
        "predicted_value": predicted_val,
        "total_shift": total_shift,
        "top_positive": top_pos,
        "top_negative": top_neg,
        "full_contributions": df_contrib.sort_values(by="abs_impact", ascending=False).reset_index(drop=True),
        "plain_english_summary": plain_english,
    }


def generate_and_save_all_figures() -> Dict[str, Path]:
    """Batch generate and persist all Stage 5 explainability figures to reports/figures/.

    Returns:
        Dictionary of saved figure paths.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    saved_paths = {}

    test_csv_path = Path("data/processed/test.csv")
    if not test_csv_path.exists():
        raise FileNotFoundError(f"Test CSV not found at {test_csv_path}.")
    test_df = pd.read_csv(test_csv_path)

    # 1. Scenario A Classification
    explainer_a, prep_a, names_a = get_cached_explainer("A", "classification")
    X_a, _, _, _, _ = prepare_scenario_data(test_df, scenario="A")
    X_trans_a = prep_a.transform(X_a)

    p_a_summary = FIGURES_DIR / "shap_summary_scenario_a.png"
    summary_plot(explainer_a, X_trans_a, names_a, is_classifier=True, title="Scenario A (Early-Warning): SHAP Beeswarm Summary", save_path=p_a_summary)
    saved_paths["shap_summary_scenario_a"] = p_a_summary

    p_a_bar = FIGURES_DIR / "shap_bar_scenario_a.png"
    bar_plot(explainer_a, X_trans_a, names_a, is_classifier=True, title="Scenario A: Global Feature Importance (|SHAP|)", save_path=p_a_bar)
    saved_paths["shap_bar_scenario_a"] = p_a_bar

    # 2. Scenario B Classification
    explainer_b, prep_b, names_b = get_cached_explainer("B", "classification")
    X_b, _, _, _, _ = prepare_scenario_data(test_df, scenario="B")
    X_trans_b = prep_b.transform(X_b)

    p_b_summary = FIGURES_DIR / "shap_summary_scenario_b.png"
    summary_plot(explainer_b, X_trans_b, names_b, is_classifier=True, title="Scenario B (Mid-Semester): SHAP Beeswarm Summary", save_path=p_b_summary)
    saved_paths["shap_summary_scenario_b"] = p_b_summary

    p_b_bar = FIGURES_DIR / "shap_bar_scenario_b.png"
    bar_plot(explainer_b, X_trans_b, names_b, is_classifier=True, title="Scenario B: Global Feature Importance (|SHAP|)", save_path=p_b_bar)
    saved_paths["shap_bar_scenario_b"] = p_b_bar

    # 3. Local Waterfalls for Scenario B (Find one at-risk and one passing student)
    at_risk_indices = test_df.index[test_df["at_risk"] == 1].tolist()
    passing_indices = test_df.index[test_df["at_risk"] == 0].tolist()

    if at_risk_indices:
        idx_risk = at_risk_indices[0]
        p_wf_risk = FIGURES_DIR / "shap_waterfall_at_risk_scenario_b.png"
        waterfall(explainer_b, X_trans_b, row_idx=idx_risk, feature_names=names_b, is_classifier=True, title=f"Student #{idx_risk} (At-Risk): SHAP Waterfall Decision Profile", save_path=p_wf_risk)
        saved_paths["shap_waterfall_at_risk_scenario_b"] = p_wf_risk

    if passing_indices:
        idx_pass = passing_indices[0]
        p_wf_pass = FIGURES_DIR / "shap_waterfall_passing_scenario_b.png"
        waterfall(explainer_b, X_trans_b, row_idx=idx_pass, feature_names=names_b, is_classifier=True, title=f"Student #{idx_pass} (Passing): SHAP Waterfall Decision Profile", save_path=p_wf_pass)
        saved_paths["shap_waterfall_passing_scenario_b"] = p_wf_pass

    return saved_paths


if __name__ == "__main__":
    print("Generating Stage 5 SHAP figures...")
    paths = generate_and_save_all_figures()
    for name, p in paths.items():
        print(f"Saved {name} -> {p}")
    print("SHAP explainability pipeline executed successfully.")
