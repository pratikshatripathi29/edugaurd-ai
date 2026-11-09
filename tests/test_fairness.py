"""
Tests for EduGuard AI Algorithmic Fairness & Bias Audit Module (Stage 5)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.config import SENSITIVE_FEATURES
from src.fairness import (
    audit_subgroup_fairness,
    comprehensive_fairness_audit,
    interpret_fairness_audit,
    plot_fairness_subgroup_metrics,
    run_full_fairness_audit,
)


@pytest.fixture
def test_data():
    """Load held-out test data."""
    return pd.read_csv("data/processed/test.csv")


def test_audit_subgroup_fairness_computes_correct_metrics():
    """Verify recall, precision, and FPR math on synthetic known cohort."""
    df_synthetic = pd.DataFrame({
        "sex": ["F", "F", "F", "F", "M", "M", "M", "M"],
    })
    y_true = np.array([1, 1, 0, 0, 1, 0, 0, 0])
    y_pred = np.array([1, 0, 1, 0, 1, 0, 0, 0])

    # For F:
    # TP=1, FN=1 (pos=2) -> Recall = 1/2 = 0.50
    # FP=1, TN=1 (neg=2) -> Precision = 1/(1+1) = 0.50, FPR = 1/2 = 0.50
    # For M:
    # TP=1, FN=0 (pos=1) -> Recall = 1/1 = 1.00
    # FP=0, TN=3 (neg=3) -> Precision = 1/1 = 1.00, FPR = 0/3 = 0.00
    res = audit_subgroup_fairness(df_synthetic, y_true, y_pred, sensitive_col="sex")

    assert len(res) == 2
    row_f = res[res["subgroup"] == "F"].iloc[0]
    row_m = res[res["subgroup"] == "M"].iloc[0]

    assert row_f["recall"] == 0.50
    assert row_f["precision"] == 0.50
    assert row_f["false_positive_rate"] == 0.50

    assert row_m["recall"] == 1.00
    assert row_m["precision"] == 1.00
    assert row_m["false_positive_rate"] == 0.00


def test_sensitive_features_not_model_inputs(test_data):
    """Verify comprehensive_fairness_audit rejects model features containing sensitive attributes."""
    y_true = test_data["at_risk"].values
    y_pred = np.zeros_like(y_true)

    # Valid call (features_used does not contain sensitive attributes)
    valid_features = ["studytime", "failures", "absences"]
    audit = comprehensive_fairness_audit(
        test_data, y_true, y_pred, sensitive_cols=["sex"], features_used=valid_features
    )
    assert not audit.empty

    # Illegal call: sensitive feature 'sex' passed in model features
    illegal_features = ["studytime", "sex", "failures"]
    with pytest.raises(ValueError, match="ETHICAL GOVERNANCE VIOLATION"):
        comprehensive_fairness_audit(
            test_data, y_true, y_pred, sensitive_cols=["sex"], features_used=illegal_features
        )


def test_subgroup_sample_size_warning(test_data):
    """Verify small groups (e.g. Rural address or MS school) trigger small_sample_warning."""
    y_true = test_data["at_risk"].values
    y_pred = np.zeros_like(y_true)

    audit_addr = audit_subgroup_fairness(test_data, y_true, y_pred, sensitive_col="address")
    row_r = audit_addr[audit_addr["subgroup"] == "R"].iloc[0]
    row_u = audit_addr[audit_addr["subgroup"] == "U"].iloc[0]

    # In test.csv, R has N=50, pos=16 (<60 and <20)
    assert row_r["small_sample_warning"] == True
    assert row_u["small_sample_warning"] == False


def test_comprehensive_fairness_audit_covers_all_target_attributes(test_data):
    """Verify audit covers sex, address, and school."""
    y_true = test_data["at_risk"].values
    y_pred = np.zeros_like(y_true)

    audit = comprehensive_fairness_audit(test_data, y_true, y_pred, sensitive_cols=["sex", "address", "school"])
    unique_attrs = set(audit["attribute"].unique())

    assert unique_attrs == {"sex", "address", "school"}
    assert "recall" in audit.columns
    assert "precision" in audit.columns
    assert "false_positive_rate" in audit.columns


def test_interpret_fairness_audit_generates_educational_narrative(test_data):
    """Verify interpretation helper provides clear text with sample size notes."""
    y_true = test_data["at_risk"].values
    y_pred = (y_true == 1).astype(int)  # perfect predictions

    audit = audit_subgroup_fairness(test_data, y_true, y_pred, sensitive_col="address")
    narrative = interpret_fairness_audit(audit, "address")

    assert "Equal Opportunity (Recall)" in narrative
    assert "Predictive Parity (Precision)" in narrative
    assert "False Alarm Parity (FPR)" in narrative
    assert "Statistical Caution" in narrative


def test_run_full_fairness_audit_persists_artifacts():
    """Verify run_full_fairness_audit saves report CSV and figure."""
    df, narrative = run_full_fairness_audit(scenario="B", threshold=0.15)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 6  # 2 sex, 2 address, 2 school
    assert len(narrative) > 200

    csv_path = Path("reports/fairness_audit_scenario_b.csv")
    fig_path = Path("reports/figures/fairness_metrics_scenario_b.png")

    assert csv_path.exists(), f"Missing CSV: {csv_path}"
    assert fig_path.exists(), f"Missing Figure: {fig_path}"

