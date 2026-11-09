"""Unit Test Suite for EduGuard AI Classification Predictions & Risk Categorization (Stage 4).

Verifies:
1. Model loading: `load_classification_model()` retrieves calibrated pipelines and metadata.
2. Prediction shape: output arrays match the number of input student records.
3. Probability bounds: all probabilities lie strictly within [0.0, 1.0].
4. Risk categorization: `categorize_risk()` correctly maps probabilities to Low, Medium, High tiers.
5. Production resilience: handles unseen categorical levels via OneHotEncoder(handle_unknown='ignore').
6. Missing value tolerance: SimpleImputer handles missing values in predictors without runtime crashes.
7. Leakage prevention: passing G3 or target flags triggers ValueError.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.config import PROCESSED_DATA_DIR, RISK_THRESHOLDS, TARGET_COL
from src.classification import (
    categorize_risk,
    load_classification_model,
    predict_risk,
    validate_classification_features,
)


@pytest.fixture(scope="module")
def sample_test_features() -> pd.DataFrame:
    """Fixture providing clean test features without targets."""
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")
    return test_df.drop(columns=[TARGET_COL, "at_risk"], errors="ignore")


# ==============================================================================
# Test 1: Model Loading
# ==============================================================================
def test_load_classification_models():
    """Verify saved calibrated classification pipelines and metadata exist and load."""
    for scenario in ["A", "B"]:
        model, metadata = load_classification_model(scenario=scenario)
        assert model is not None
        assert isinstance(metadata, dict)
        assert "features_used" in metadata
        assert "optimal_threshold" in metadata
        assert "test_metrics_default_threshold" in metadata
        assert "test_metrics_optimal_threshold" in metadata


def test_load_classification_missing_file_raises(tmp_path: Path):
    """Verify FileNotFoundError is raised when an invalid model path is provided."""
    fake_path = tmp_path / "non_existent_classifier.joblib"
    with pytest.raises(FileNotFoundError, match="Classification model file not found"):
        load_classification_model(scenario="A", model_path=fake_path)


# ==============================================================================
# Test 2: Prediction Output Shapes
# ==============================================================================
def test_prediction_shapes(sample_test_features: pd.DataFrame):
    """Verify output dictionary arrays match the length of candidate inputs."""
    for scenario in ["A", "B"]:
        model, metadata = load_classification_model(scenario=scenario)
        result = predict_risk(model, metadata, sample_test_features)

        n_samples = len(sample_test_features)
        assert len(result["probabilities"]) == n_samples
        assert len(result["predictions"]) == n_samples
        assert len(result["risk_tiers"]) == n_samples
        assert "threshold_used" in result


# ==============================================================================
# Test 3: Probability Range Bounds [0.0, 1.0]
# ==============================================================================
def test_probabilities_in_valid_range(sample_test_features: pd.DataFrame):
    """Verify that all output probabilities are valid statistical probabilities."""
    for scenario in ["A", "B"]:
        model, metadata = load_classification_model(scenario=scenario)
        result = predict_risk(model, metadata, sample_test_features)
        probs = result["probabilities"]

        assert not np.isnan(probs).any(), "NaN found in predicted probabilities"
        assert (probs >= 0.0).all(), "Found negative probabilities"
        assert (probs <= 1.0).all(), "Found probabilities exceeding 1.0"


# ==============================================================================
# Test 4: Categorize Risk Boundaries
# ==============================================================================
def test_categorize_risk_boundaries_default_thresholds():
    """Verify risk tier mapping under default RISK_THRESHOLDS (low < 0.30, high > 0.60)."""
    # Scalar tests
    assert categorize_risk(0.0) == "Low"
    assert categorize_risk(0.299) == "Low"
    assert categorize_risk(0.30) == "Medium"  # Boundary: low_thresh is inclusive for Medium
    assert categorize_risk(0.45) == "Medium"
    assert categorize_risk(0.60) == "Medium"  # Boundary: high_thresh is inclusive for Medium
    assert categorize_risk(0.601) == "High"
    assert categorize_risk(1.0) == "High"

    # Array / Vectorized tests
    test_probs = np.array([0.10, 0.30, 0.50, 0.60, 0.85])
    expected_tiers = np.array(["Low", "Medium", "Medium", "Medium", "High"])
    actual_tiers = categorize_risk(test_probs)
    np.testing.assert_array_equal(actual_tiers, expected_tiers)


def test_categorize_risk_custom_thresholds():
    """Verify risk categorization supports custom configurable thresholds."""
    custom_thresholds = {"low": 0.20, "high": 0.50}

    assert categorize_risk(0.15, thresholds=custom_thresholds) == "Low"
    assert categorize_risk(0.20, thresholds=custom_thresholds) == "Medium"
    assert categorize_risk(0.35, thresholds=custom_thresholds) == "Medium"
    assert categorize_risk(0.50, thresholds=custom_thresholds) == "Medium"
    assert categorize_risk(0.51, thresholds=custom_thresholds) == "High"


# ==============================================================================
# Test 5: Unseen Categories Resilience
# ==============================================================================
def test_unseen_categories_resilience(sample_test_features: pd.DataFrame):
    """Verify pipeline gracefully handles new/unseen categories without crashing."""
    dirty_candidate = sample_test_features.iloc[[0]].copy()
    dirty_candidate.loc[dirty_candidate.index[0], "school"] = "UNSEEN_ACADEMY_99"
    dirty_candidate.loc[dirty_candidate.index[0], "reason"] = "SPACE_MISSION"
    dirty_candidate.loc[dirty_candidate.index[0], "subject"] = "ROBOTICS"

    for scenario in ["A", "B"]:
        model, metadata = load_classification_model(scenario=scenario)
        # Should execute cleanly without raising ValueError
        result = predict_risk(model, metadata, dirty_candidate)
        assert len(result["probabilities"]) == 1
        assert 0.0 <= result["probabilities"][0] <= 1.0


# ==============================================================================
# Test 6: Missing Values Tolerance
# ==============================================================================
def test_missing_values_tolerance(sample_test_features: pd.DataFrame):
    """Verify preprocessor imputes missing numerical and categorical values in inference."""
    nan_candidate = sample_test_features.iloc[[0]].copy()
    nan_candidate.loc[nan_candidate.index[0], "absences"] = np.nan
    nan_candidate.loc[nan_candidate.index[0], "studytime"] = np.nan
    nan_candidate.loc[nan_candidate.index[0], "reason"] = np.nan

    for scenario in ["A", "B"]:
        model, metadata = load_classification_model(scenario=scenario)
        result = predict_risk(model, metadata, nan_candidate)
        assert not np.isnan(result["probabilities"]).any()
        assert result["risk_tiers"][0] in ["Low", "Medium", "High"]


# ==============================================================================
# Test 7: Target Leakage Rejection
# ==============================================================================
def test_inference_rejects_leakage():
    """Verify candidate inputs containing G3 or at_risk are rejected with ValueError."""
    leaking_df = pd.DataFrame(
        {
            "school": ["GP"],
            "G3": [14.0],  # Leakage
            "at_risk": [0], # Leakage
        }
    )
    _, metadata = load_classification_model(scenario="A")

    with pytest.raises(ValueError, match="DATA LEAKAGE ERROR"):
        validate_classification_features(leaking_df, metadata)

