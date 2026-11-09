"""
Tests for EduGuard AI Explainability Module (Stage 5)
"""

import numpy as np
import pandas as pd
import pytest

from src.config import RANDOM_STATE
from src.explainability import (
    clear_explainer_cache,
    compute_shap_explanation,
    explain_student,
    get_cached_explainer,
    get_output_scale_info,
    get_transformed_features,
    global_importance,
    unwrap_pipeline_and_tree,
)
from src.features import prepare_scenario_data


@pytest.fixture(autouse=True)
def reset_cache():
    """Ensure clean explainer cache for each test."""
    clear_explainer_cache()
    yield
    clear_explainer_cache()


@pytest.fixture
def test_dataset():
    """Load held-out test data."""
    return pd.read_csv("data/processed/test.csv")


def test_explainer_caching():
    """Verify get_cached_explainer memoizes explainers."""
    exp1, prep1, names1 = get_cached_explainer("A", "classification")
    exp2, prep2, names2 = get_cached_explainer("A", "classification")

    assert exp1 is exp2, "Cached explainer instance must be reused."
    assert prep1 is prep2, "Cached preprocessor instance must be reused."
    assert names1 == names2, "Feature names must match exactly."


def test_feature_names_match_preprocessor(test_dataset):
    """Verify transformed feature names match preprocessor output schema."""
    _, prep, names = get_cached_explainer("A", "classification")
    X_raw, _, _, _, _ = prepare_scenario_data(test_dataset, scenario="A")
    X_trans, extracted_names = get_transformed_features(prep, X_raw)

    assert X_trans.shape[1] == len(names)
    assert extracted_names == names
    assert "failures" in names
    assert "studytime" in names
    assert "absences" in names


def test_shap_additivity_property(test_dataset):
    """Verify Shapley local accuracy: base_value + sum(shap_values) == model output."""
    explainer, prep, names = get_cached_explainer("A", "classification")
    X_raw, _, _, _, _ = prepare_scenario_data(test_dataset.head(5), scenario="A")
    X_trans, _ = get_transformed_features(prep, X_raw)

    exp = compute_shap_explanation(explainer, X_trans, names, is_classifier=True)

    # In RandomForestClassifier, TreeExplainer outputs probabilities for class 1
    # Check that sum of SHAP values + base_value matches tree model's predict_proba
    from src.classification import load_classification_model
    clf, _ = load_classification_model("A")
    _, tree_model, _ = unwrap_pipeline_and_tree(clf)
    rf_probs = tree_model.predict_proba(X_trans)[:, 1]

    for i in range(len(X_trans)):
        shap_sum = float(np.sum(exp[i].values))
        base_val = float(np.squeeze(exp[i].base_values))
        reconstructed = base_val + shap_sum

        assert np.isclose(reconstructed, rf_probs[i], atol=1e-5), (
            f"Additivity failed for sample {i}: reconstructed={reconstructed}, actual={rf_probs[i]}"
        )


def test_explain_student_positive_and_negative_contributions(test_dataset):
    """Verify explain_student generates valid positive and negative contribution DataFrames."""
    explainer, prep, names = get_cached_explainer("A", "classification")
    student_row = test_dataset.iloc[0]

    result = explain_student(student_row, explainer, prep, feature_names=names, top_n=3)

    assert "base_value" in result
    assert "predicted_value" in result
    assert "top_positive" in result
    assert "top_negative" in result
    assert "plain_english_summary" in result

    top_pos = result["top_positive"]
    top_neg = result["top_negative"]

    assert isinstance(top_pos, pd.DataFrame)
    assert isinstance(top_neg, pd.DataFrame)
    assert len(top_pos) <= 3
    assert len(top_neg) <= 3

    # Positive SHAP values elevate risk
    if not top_pos.empty:
        assert (top_pos["shap_value"] > 0).all()
        assert (top_pos["direction"] == "Elevates Risk").all()

    # Negative SHAP values mitigate risk
    if not top_neg.empty:
        assert (top_neg["shap_value"] < 0).all()
        assert (top_neg["direction"] == "Mitigates Risk").all()


def test_explain_student_plain_english_uses_real_values(test_dataset):
    """Verify generated plain-English summary quotes real calculated numbers."""
    explainer, prep, names = get_cached_explainer("A", "classification")
    student_row = test_dataset.iloc[0]

    result = explain_student(student_row, explainer, prep, feature_names=names, top_n=3)
    summary = result["plain_english_summary"]

    assert isinstance(summary, str)
    assert len(summary) > 50

    # Ensure real percentages appear in the text
    base_pct_str = f"{result['base_value'] * 100:.1f}%"
    pred_pct_str = f"{result['predicted_value'] * 100:.1f}%"

    assert base_pct_str in summary, f"Base percentage {base_pct_str} missing from summary."
    assert pred_pct_str in summary, f"Predicted percentage {pred_pct_str} missing from summary."
    assert "Advisory Note" in summary


def test_explainability_rejects_leakage(test_dataset):
    """Verify get_transformed_features rejects target leakage when auto_drop_leakage is False."""
    _, prep, _ = get_cached_explainer("A", "classification")

    # test_dataset contains 'G3' and 'at_risk'
    with pytest.raises(ValueError, match="TARGET LEAKAGE DETECTED"):
        get_transformed_features(prep, test_dataset.head(5), auto_drop_leakage=False)


def test_global_importance_sorting(test_dataset):
    """Verify global_importance returns a sorted DataFrame with descending mean absolute SHAP."""
    explainer, prep, names = get_cached_explainer("A", "classification")
    X_raw, _, _, _, _ = prepare_scenario_data(test_dataset.head(20), scenario="A")
    X_trans, _ = get_transformed_features(prep, X_raw)

    df_imp = global_importance(explainer, X_trans, feature_names=names)

    assert isinstance(df_imp, pd.DataFrame)
    assert list(df_imp.columns) == ["feature", "mean_abs_shap"]
    assert (df_imp["mean_abs_shap"] >= 0).all()

    # Verify descending sort order
    assert df_imp["mean_abs_shap"].is_monotonic_decreasing


def test_output_scale_info():
    """Verify output scale explanation helper accurately identifies scales."""
    explainer_clf, _, _ = get_cached_explainer("A", "classification")
    scale_info_clf = get_output_scale_info(explainer_clf.model, is_classifier=True)
    assert scale_info_clf["scale"] == "probability"
    assert "predicted probability" in scale_info_clf["unit"].lower()

    explainer_reg, _, _ = get_cached_explainer("A", "regression")
    scale_info_reg = get_output_scale_info(explainer_reg.model, is_classifier=False)
    assert scale_info_reg["scale"] == "grade_points"
    assert "final grade points" in scale_info_reg["unit"].lower()
