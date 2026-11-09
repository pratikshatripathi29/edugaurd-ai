"""EduGuard AI - Feature Engineering and Scenario Construction Module (Stage 2).

This module defines:
1. Leakage-free engineered features with clear educational rationales.
2. Distinct feature sets for Scenario A (early-warning, no G1/G2) and Scenario B (mid-semester, with G1/G2).
3. Target extraction for regression (G3) and classification (at_risk: G3 < 10).
4. Strict enforcement of algorithmic fairness: sensitive attributes are NEVER in feature sets.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.config import (
    INTERIM_GRADE_COLS,
    LEAKAGE_COLS,
    RISK_PASS_MARK,
    SENSITIVE_FEATURES,
    TARGET_COL,
)


def add_engineered_features(df: pd.DataFrame, include_trend: bool = False) -> pd.DataFrame:
    """Generate leakage-safe, pedagogically grounded engineered features.

    Engineered Features & Educational Rationale:
    -------------------------------------------
    1. alcohol_total (Dalc + Walc):
       Combines workday and weekend alcohol consumption into an aggregate scale (2 - 10).
       Excessive alcohol consumption is strongly correlated with absenteeism and academic disengagement.
    2. support_total:
       Sum of active academic assistance flags (schoolsup='yes' + famsup='yes' + paid='yes').
       Measures the total intervention support a student is already receiving.
    3. high_absence:
       Binary indicator for chronic absenteeism (absences >= 10).
       In educational research, missing >= 10 sessions is a reliable early trigger for academic peril.
    4. study_time_ratio:
       Ratio of weekly study time to free time (studytime / (freetime + 1.0)).
       Captures commitment balance: high values signal focused academic discipline; low values signal leisure imbalance.
    5. failures_flag:
       Binary indicator for previous course failures (failures > 0).
       Prior academic setbacks represent one of the single most reliable predictors of future risk.
    6. grade_trend (G2 - G1) [SCENARIO B ONLY]:
       Academic momentum between Period 1 and Period 2.
       Negative values signal deteriorating grades requiring intervention; positive values reflect upward recovery.

    Args:
        df: Input DataFrame.
        include_trend: If True and G1/G2 are present, computes grade_trend for Scenario B.

    Returns:
        DataFrame copy with new feature columns appended.
    """
    feat_df = df.copy()

    # 1. Total alcohol consumption
    if "Dalc" in feat_df.columns and "Walc" in feat_df.columns:
        feat_df["alcohol_total"] = feat_df["Dalc"] + feat_df["Walc"]

    # 2. Total support interventions
    support_cols = ["schoolsup", "famsup", "paid"]
    if all(col in feat_df.columns for col in support_cols):
        feat_df["support_total"] = (
            (feat_df["schoolsup"] == "yes").astype(int)
            + (feat_df["famsup"] == "yes").astype(int)
            + (feat_df["paid"] == "yes").astype(int)
        )

    # 3. High absence / chronic absenteeism flag
    if "absences" in feat_df.columns:
        feat_df["high_absence"] = (feat_df["absences"] >= 10).astype(int)

    # 4. Study-to-free-time ratio
    if "studytime" in feat_df.columns and "freetime" in feat_df.columns:
        feat_df["study_time_ratio"] = feat_df["studytime"] / (feat_df["freetime"] + 1.0)

    # 5. Prior failures indicator flag
    if "failures" in feat_df.columns:
        feat_df["failures_flag"] = (feat_df["failures"] > 0).astype(int)

    # 6. Scenario B momentum trend (G2 - G1)
    if include_trend and "G1" in feat_df.columns and "G2" in feat_df.columns:
        feat_df["grade_trend"] = feat_df["G2"] - feat_df["G1"]

    return feat_df


def get_feature_sets() -> Dict[str, Dict[str, List[str]]]:
    """Return feature column specifications for Scenario A and Scenario B.

    GOVERNANCE RULES ENFORCED:
    - Never include G3 (target variable).
    - Never include sensitive attributes from SENSITIVE_FEATURES (sex, age, address,
      famsize, Pstatus, Medu, Fedu, Mjob, Fjob, guardian, nationality).
    - Never include student_key or leakage targets (passed, at_risk).
    - Scenario A strictly omits G1, G2, and grade_trend.
    - Scenario B includes Scenario A features plus G1, G2, and grade_trend.
    - 'subject' is included in both scenarios to identify course context ('math' vs. 'portuguese').

    Returns:
        Dictionary containing 'scenario_a' and 'scenario_b' feature dictionaries:
        {
            "scenario_a": {"numeric": [...], "categorical": [...], "all": [...]},
            "scenario_b": {"numeric": [...], "categorical": [...], "all": [...]}
        }
    """
    # Base non-sensitive categorical features
    cat_features = [
        "school",
        "reason",
        "schoolsup",
        "famsup",
        "paid",
        "activities",
        "nursery",
        "higher",
        "internet",
        "romantic",
        "subject",  # 'math' or 'portuguese'
    ]

    # Base non-sensitive numeric features for Scenario A (Early Warning)
    num_features_a = [
        "traveltime",
        "studytime",
        "failures",
        "famrel",
        "freetime",
        "goout",
        "Dalc",
        "Walc",
        "health",
        "absences",
        # Engineered non-leaking features
        "alcohol_total",
        "support_total",
        "high_absence",
        "study_time_ratio",
        "failures_flag",
    ]

    # Scenario B includes Scenario A plus interim grades and trend
    num_features_b = num_features_a + [
        "G1",
        "G2",
        "grade_trend",
    ]

    # Verify zero intersection with sensitive features
    for col in cat_features + num_features_b:
        assert col not in SENSITIVE_FEATURES, f"FAIRNESS VIOLATION: '{col}' is in SENSITIVE_FEATURES!"
        assert col not in LEAKAGE_COLS, f"LEAKAGE VIOLATION: '{col}' is in LEAKAGE_COLS!"
        assert col != "student_key", "LEAKAGE VIOLATION: 'student_key' cannot be a feature!"

    return {
        "scenario_a": {
            "numeric": num_features_a,
            "categorical": cat_features,
            "all": num_features_a + cat_features,
        },
        "scenario_b": {
            "numeric": num_features_b,
            "categorical": cat_features,
            "all": num_features_b + cat_features,
        },
    }


def make_targets(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Extract regression target (G3) and operational classification target (at_risk).

    Args:
        df: Input DataFrame containing the 'G3' final grade column.

    Returns:
        Tuple of:
        - y_reg: Continuous final grade G3 (0 - 20 scale).
        - y_clf: Binary at-risk label (1 if G3 < RISK_PASS_MARK else 0).

    Raises:
        KeyError: If G3 is not present in the input DataFrame.
    """
    if TARGET_COL not in df.columns:
        raise KeyError(f"Target column '{TARGET_COL}' not found in DataFrame.")

    y_reg = df[TARGET_COL].astype(float).copy()
    y_clf = (df[TARGET_COL] < RISK_PASS_MARK).astype(int).copy()

    return y_reg, y_clf


def prepare_scenario_data(
    df: pd.DataFrame,
    scenario: str = "A",
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, List[str], List[str]]:
    """Helper to prepare feature matrix X, targets y_reg, y_clf, and column lists for a scenario.

    Args:
        df: Cleaned DataFrame.
        scenario: 'A' for early-warning or 'B' for mid-semester.

    Returns:
        Tuple of (X, y_reg, y_clf, numeric_cols, categorical_cols).
    """
    scenario_key = "scenario_a" if scenario.upper() == "A" else "scenario_b"
    include_trend = (scenario.upper() == "B")

    # Add engineered features
    augmented_df = add_engineered_features(df, include_trend=include_trend)

    # Get feature lists
    feature_sets = get_feature_sets()
    specs = feature_sets[scenario_key]
    numeric_cols = specs["numeric"]
    categorical_cols = specs["categorical"]
    all_features = specs["all"]

    # Feature matrix X
    X = augmented_df[all_features].copy()

    # Targets
    y_reg, y_clf = make_targets(augmented_df)

    return X, y_reg, y_clf, numeric_cols, categorical_cols
