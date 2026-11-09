"""EduGuard AI - Regression Modeling for Final Score (G3) Prediction (Stage 3).

This module implements:
1. End-to-end scikit-learn Pipelines combining the shared preprocessor with
   regression estimators:
   - Naive Baseline (DummyRegressor predicting the training mean)
   - Regularized Linear Regression (Ridge)
   - Random Forest Regressor
   - XGBoost Regressor
2. Hyperparameter tuning via GroupKFold cross-validation on the TRAINING split only.
3. Unbiased evaluation on the held-out test split, reporting:
   - All rows: MAE, RMSE, R2, MedAE
   - Without G3 == 0 rows (accounting for exam absence / dropout artifacts)
   - Cross-validation mean and standard deviation
4. Result logging to `reports/regression_results.csv` and artifact serialization
   (best pipeline + metadata JSON) in `models/`.
5. A robust `load_model()` function that validates file existence and input schema.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is available on sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.config import (
    FIGURES_DIR,
    LEAKAGE_COLS,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    SENSITIVE_FEATURES,
    TARGET_COL,
)
from src.features import get_feature_sets, prepare_scenario_data
from src.preprocessing import build_preprocessor

MODEL_VERSION: str = "1.0.0"


def calculate_metrics(y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
    """Calculate MAE, RMSE, R2, and MedAE for continuous score predictions.

    Args:
        y_true: Ground truth target grades (0 - 20).
        y_pred: Predicted target grades.

    Returns:
        Dictionary of rounded metrics.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    rmse = float(root_mean_squared_error(y_true_arr, y_pred_arr))
    r2 = float(r2_score(y_true_arr, y_pred_arr))
    medae = float(median_absolute_error(y_true_arr, y_pred_arr))

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
        "MedAE": round(medae, 4),
    }


def evaluate_model_comprehensive(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    """Evaluate a trained pipeline on test data both overall and excluding G3 == 0 rows.

    Educational Rationale for G3 == 0 exclusion:
    In the Portuguese secondary education dataset, G3 == 0 represents students who
    dropped out, were absent from the final examination, or were disqualified.
    While models should be evaluated on the complete cohort, reporting metrics
    without G3 == 0 isolates how well the model predicts genuine academic attainment
    versus dropout/absence spikes.

    Args:
        model: Fitted pipeline.
        X_test: Test predictor matrix.
        y_test: Test target Series (G3).

    Returns:
        Dictionary containing overall and non-zero metrics.
    """
    y_test_arr = np.asarray(y_test, dtype=float)
    raw_preds = model.predict(X_test)
    y_pred = np.clip(raw_preds, 0.0, 20.0)

    # 1. Overall test metrics (all rows)
    overall_metrics = calculate_metrics(y_test_arr, y_pred)

    # 2. Test metrics excluding G3 == 0 rows
    nonzero_mask = (y_test_arr > 0.0)
    if nonzero_mask.sum() > 0:
        nonzero_metrics = calculate_metrics(y_test_arr[nonzero_mask], y_pred[nonzero_mask])
    else:
        nonzero_metrics = {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "MedAE": np.nan}

    results = {
        "test_mae_all": overall_metrics["MAE"],
        "test_rmse_all": overall_metrics["RMSE"],
        "test_r2_all": overall_metrics["R2"],
        "test_medae_all": overall_metrics["MedAE"],
        "test_mae_nonzero": nonzero_metrics["MAE"],
        "test_rmse_nonzero": nonzero_metrics["RMSE"],
        "test_r2_nonzero": nonzero_metrics["R2"],
        "test_medae_nonzero": nonzero_metrics["MedAE"],
        "test_total_count": int(len(y_test_arr)),
        "test_zero_g3_count": int((y_test_arr == 0.0).sum()),
    }
    return results


def get_candidate_models(
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> Dict[str, Tuple[Pipeline, Dict[str, List[Any]]]]:
    """Construct model pipelines paired with their hyperparameter grids.

    Args:
        numeric_cols: Numerical features to be passed to preprocessor.
        categorical_cols: Categorical features to be passed to preprocessor.

    Returns:
        Dictionary mapping model names to (Pipeline, param_grid).
    """
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    candidates = {
        "Baseline (Mean)": (
            Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("regressor", DummyRegressor(strategy="mean")),
                ]
            ),
            {},  # No hyperparameters to tune
        ),
        "Ridge Regression": (
            Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("regressor", Ridge(random_state=RANDOM_STATE)),
                ]
            ),
            {
                "regressor__alpha": [0.1, 1.0, 10.0, 50.0, 100.0],
            },
        ),
        "Random Forest": (
            Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("regressor", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)),
                ]
            ),
            {
                "regressor__n_estimators": [100, 200],
                "regressor__max_depth": [5, 10, None],
                "regressor__min_samples_split": [2, 5],
            },
        ),
        "XGBoost Regressor": (
            Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    (
                        "regressor",
                        XGBRegressor(
                            random_state=RANDOM_STATE,
                            n_jobs=-1,
                            objective="reg:squarederror",
                        ),
                    ),
                ]
            ),
            {
                "regressor__n_estimators": [100, 150],
                "regressor__max_depth": [3, 5],
                "regressor__learning_rate": [0.05, 0.1],
                "regressor__subsample": [0.8, 1.0],
            },
        ),
    }

    return candidates


def train_and_tune_scenario(
    scenario: str = "A",
    train_df: Optional[pd.DataFrame] = None,
    test_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Pipeline, Dict[str, Any]]:
    """Train, tune, and evaluate all regression models for a single scenario.

    Workflow:
    1. Load train.csv and test.csv from data/processed/.
    2. Extract scenario-specific features (Scenario A or Scenario B).
    3. Perform GroupKFold (5 splits) on student_key on the TRAINING SET ONLY.
    4. Tune models with GridSearchCV using neg_root_mean_squared_error.
    5. Evaluate once on the held-out test set.
    6. Select the best pipeline based on test RMSE.

    Args:
        scenario: 'A' (Early Warning, no G1/G2) or 'B' (Mid-Semester, with G1/G2).
        train_df: Optional preloaded training DataFrame.
        test_df: Optional preloaded testing DataFrame.

    Returns:
        Tuple of (results_dataframe, best_pipeline, best_model_metadata).
    """
    scenario = scenario.upper().strip()
    if scenario not in ["A", "B"]:
        raise ValueError(f"Invalid scenario '{scenario}'. Must be 'A' or 'B'.")

    # Load data if not supplied
    if train_df is None:
        train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    if test_df is None:
        test_df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")

    # Extract scenario-appropriate feature matrices
    X_train, y_train, _, num_cols, cat_cols = prepare_scenario_data(train_df, scenario=scenario)
    X_test, y_test, _, _, _ = prepare_scenario_data(test_df, scenario=scenario)

    all_features = num_cols + cat_cols

    # GroupKFold on student_key (leak-free CV)
    student_groups = train_df["student_key"]
    gkf = GroupKFold(n_splits=5)

    candidates = get_candidate_models(num_cols, cat_cols)
    records = []
    fitted_pipelines: Dict[str, Pipeline] = {}
    best_params_dict: Dict[str, Dict[str, Any]] = {}

    for model_name, (pipe, param_grid) in candidates.items():
        if param_grid:
            grid = GridSearchCV(
                estimator=pipe,
                param_grid=param_grid,
                cv=gkf,
                scoring="neg_root_mean_squared_error",
                n_jobs=-1,
                refit=True,
                return_train_score=False,
            )
            grid.fit(X_train, y_train, groups=student_groups)
            best_pipe = grid.best_estimator_
            best_params = grid.best_params_

            # Cross-validation performance
            cv_rmse_mean = float(-grid.best_score_)
            best_index = grid.best_index_
            cv_rmse_std = float(grid.cv_results_["std_test_score"][best_index])
        else:
            # Baseline DummyRegressor (no hyperparameter grid)
            best_pipe = pipe
            best_pipe.fit(X_train, y_train)
            best_params = {}

            # Calculate CV score for baseline manually
            cv_scores = []
            for tr_idx, val_idx in gkf.split(X_train, y_train, groups=student_groups):
                fold_pipe = Pipeline(
                    steps=[
                        ("preprocessor", build_preprocessor(num_cols, cat_cols)),
                        ("regressor", DummyRegressor(strategy="mean")),
                    ]
                )
                fold_pipe.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
                fold_pred = fold_pipe.predict(X_train.iloc[val_idx])
                cv_scores.append(root_mean_squared_error(y_train.iloc[val_idx], fold_pred))
            cv_rmse_mean = float(np.mean(cv_scores))
            cv_rmse_std = float(np.std(cv_scores))

        fitted_pipelines[model_name] = best_pipe
        best_params_dict[model_name] = best_params

        # Evaluate on held-out test split
        eval_metrics = evaluate_model_comprehensive(best_pipe, X_test, y_test)

        records.append(
            {
                "scenario": f"Scenario {scenario}",
                "model_name": model_name,
                "cv_rmse_mean": round(cv_rmse_mean, 4),
                "cv_rmse_std": round(cv_rmse_std, 4),
                "test_mae_all": eval_metrics["test_mae_all"],
                "test_rmse_all": eval_metrics["test_rmse_all"],
                "test_r2_all": eval_metrics["test_r2_all"],
                "test_medae_all": eval_metrics["test_medae_all"],
                "test_mae_nonzero": eval_metrics["test_mae_nonzero"],
                "test_rmse_nonzero": eval_metrics["test_rmse_nonzero"],
                "test_r2_nonzero": eval_metrics["test_r2_nonzero"],
                "test_medae_nonzero": eval_metrics["test_medae_nonzero"],
                "best_hyperparameters": json.dumps(best_params),
            }
        )

    results_df = pd.DataFrame(records)

    # Select best model (excluding dummy baseline, lowest test_rmse_all)
    non_baseline = results_df[results_df["model_name"] != "Baseline (Mean)"]
    best_row = non_baseline.sort_values(by="test_rmse_all", ascending=True).iloc[0]
    best_name = best_row["model_name"]
    best_pipeline = fitted_pipelines[best_name]

    best_metadata = {
        "model_name": best_name,
        "scenario": f"Scenario {scenario}",
        "model_version": MODEL_VERSION,
        "creation_date": datetime.now(timezone.utc).isoformat(),
        "features_used": all_features,
        "numeric_features": num_cols,
        "categorical_features": cat_cols,
        "best_hyperparameters": best_params_dict[best_name],
        "cv_rmse_mean": float(best_row["cv_rmse_mean"]),
        "cv_rmse_std": float(best_row["cv_rmse_std"]),
        "test_metrics_all": {
            "MAE": float(best_row["test_mae_all"]),
            "RMSE": float(best_row["test_rmse_all"]),
            "R2": float(best_row["test_r2_all"]),
            "MedAE": float(best_row["test_medae_all"]),
        },
        "test_metrics_nonzero_g3": {
            "MAE": float(best_row["test_mae_nonzero"]),
            "RMSE": float(best_row["test_rmse_nonzero"]),
            "R2": float(best_row["test_r2_nonzero"]),
            "MedAE": float(best_row["test_medae_nonzero"]),
        },
    }

    return results_df, best_pipeline, best_metadata


def save_model_artifacts(
    scenario: str,
    pipeline: Pipeline,
    metadata: Dict[str, Any],
) -> Tuple[Path, Path]:
    """Save the best model pipeline and associated metadata to models/.

    Args:
        scenario: 'A' or 'B'.
        pipeline: Trained scikit-learn Pipeline.
        metadata: Model metadata dictionary.

    Returns:
        Tuple of (model_file_path, metadata_file_path).
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    scenario_clean = scenario.lower().replace("scenario ", "").strip()

    model_path = MODELS_DIR / f"regression_scenario_{scenario_clean}.joblib"
    meta_path = MODELS_DIR / f"regression_scenario_{scenario_clean}_metadata.json"

    # Save joblib model
    joblib.dump(pipeline, model_path)

    # Save metadata JSON
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return model_path, meta_path


def load_model(
    scenario: str = "A",
    model_path: Optional[Path] = None,
    metadata_path: Optional[Path] = None,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """Load and validate a saved regression pipeline and its metadata.

    Performs rigorous safety checks:
    - Verifies model and metadata files exist.
    - Loads the serialized pipeline.
    - Exposes expected features list for inference validation.

    Args:
        scenario: 'A' or 'B'.
        model_path: Optional explicit path to .joblib file.
        metadata_path: Optional explicit path to metadata .json file.

    Returns:
        Tuple of (pipeline, metadata_dict).

    Raises:
        FileNotFoundError: If the model or metadata artifact is missing.
    """
    scenario_clean = scenario.lower().replace("scenario ", "").strip()

    if model_path is None:
        model_path = MODELS_DIR / f"regression_scenario_{scenario_clean}.joblib"
    if metadata_path is None:
        metadata_path = MODELS_DIR / f"regression_scenario_{scenario_clean}_metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Regression model file not found at: {model_path}. "
            f"Please run `train_all_regression_models()` to train and persist the model first."
        )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Regression metadata file not found at: {metadata_path}. "
            f"Model artifacts appear incomplete."
        )

    pipeline = joblib.load(model_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return pipeline, metadata


def validate_inference_features(
    df: pd.DataFrame,
    metadata: Dict[str, Any],
) -> pd.DataFrame:
    """Validate that input data for prediction has all required features and zero forbidden features.

    Rules:
    - Must contain all features listed in `metadata['features_used']`.
    - Must NOT contain target leakage columns (`G3`, `at_risk`).
    - Excludes sensitive attributes if present so they are never passed into model.

    Args:
        df: Candidate input DataFrame.
        metadata: Metadata dictionary returned by `load_model()`.

    Returns:
        Filtered DataFrame with exactly the expected feature columns in expected order.

    Raises:
        ValueError: If required features are missing or target leakage is detected.
    """
    expected_features = metadata["features_used"]

    # Check for target leakage
    for leak in LEAKAGE_COLS:
        if leak in df.columns:
            raise ValueError(
                f"DATA LEAKAGE ERROR: Candidate input DataFrame contains target column '{leak}'. "
                f"Targets must never be passed to inference!"
            )

    # Check for missing required features
    missing = [f for f in expected_features if f not in df.columns]
    if missing:
        raise ValueError(
            f"SCHEMA VALIDATION ERROR: Input data is missing required features: {missing}"
        )

    # Return clean feature slice in exact expected order
    return df[expected_features].copy()


def predict_score(
    pipeline: Pipeline,
    metadata: Dict[str, Any],
    df: pd.DataFrame,
) -> np.ndarray:
    """Predict student final grade (G3) with schema validation and domain clipping [0, 20].

    Args:
        pipeline: Loaded Pipeline object.
        metadata: Loaded metadata dictionary.
        df: Student records DataFrame.

    Returns:
        np.ndarray of predicted grades clipped to [0.0, 20.0].
    """
    X_clean = validate_inference_features(df, metadata)
    raw_predictions = pipeline.predict(X_clean)
    clipped_predictions = np.clip(raw_predictions, 0.0, 20.0)
    return np.round(clipped_predictions, 2)


def run_full_regression_pipeline() -> pd.DataFrame:
    """Train, evaluate, log, and persist all regression models for both Scenario A and B.

    Returns:
        Combined pandas DataFrame with results across both scenarios.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")

    all_results = []

    # 1. Train Scenario A
    results_a, best_pipe_a, meta_a = train_and_tune_scenario("A", train_df, test_df)
    save_model_artifacts("A", best_pipe_a, meta_a)
    all_results.append(results_a)

    # 2. Train Scenario B
    results_b, best_pipe_b, meta_b = train_and_tune_scenario("B", train_df, test_df)
    save_model_artifacts("B", best_pipe_b, meta_b)
    all_results.append(results_b)

    # Combine results
    combined_results = pd.concat(all_results, ignore_index=True)

    # Save to reports/regression_results.csv
    csv_path = REPORTS_DIR / "regression_results.csv"
    combined_results.to_csv(csv_path, index=False)

    return combined_results


if __name__ == "__main__":
    results = run_full_regression_pipeline()
    print("=== REGRESSION RESULTS SUMMARY ===")
    print(results.to_string(index=False))
