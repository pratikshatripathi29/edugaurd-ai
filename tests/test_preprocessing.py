"""Unit Test Suite for EduGuard AI Preprocessing, Feature Engineering & Governance.

Verifies:
1. Zero student identity leakage in GroupShuffleSplit (no overlapping groups).
2. G3 target and derived targets are NEVER in any feature set.
3. Sensitive/protected demographic attributes are strictly excluded from feature sets.
4. Unseen categorical levels at test/inference time do not crash the preprocessor.
5. The `clean()` function repairs invalid ranges, drops duplicates, and returns an audit report.
6. Target generation accurately enforces RISK_PASS_MARK (10.0).
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from src.config import (
    LEAKAGE_COLS,
    RANDOM_STATE,
    RISK_PASS_MARK,
    SENSITIVE_FEATURES,
    TARGET_COL,
)
from src.data_loader import load_raw_dataset
from src.features import (
    add_engineered_features,
    get_feature_sets,
    make_targets,
    prepare_scenario_data,
)
from src.preprocessing import (
    build_preprocessor,
    clean,
    grouped_train_test_split,
    make_student_key,
)


@pytest.fixture(scope="module")
def combined_raw_data() -> pd.DataFrame:
    """Fixture providing raw combined Math and Portuguese datasets."""
    mat = load_raw_dataset("mat").copy()
    por = load_raw_dataset("por").copy()
    mat["subject"] = "math"
    por["subject"] = "portuguese"
    df = pd.concat([mat, por], ignore_index=True)
    df["is_simulated"] = False
    return df


# ==============================================================================
# Test 1: Zero Overlapping Student Groups
# ==============================================================================
def test_grouped_split_no_overlapping_groups(combined_raw_data: pd.DataFrame):
    """Verify GroupShuffleSplit completely prevents student_key overlap across train and test."""
    df = combined_raw_data.copy()
    df["student_key"] = make_student_key(df)

    train_df, test_df = grouped_train_test_split(
        df, test_size=0.2, random_state=RANDOM_STATE, group_col="student_key"
    )

    train_keys = set(train_df["student_key"])
    test_keys = set(test_df["student_key"])

    overlapping_students = train_keys.intersection(test_keys)

    # Core assertion: zero students appear on both sides of the split
    assert len(overlapping_students) == 0, (
        f"Data leakage detected! {len(overlapping_students)} students appear in both "
        f"train ({len(train_keys)}) and test ({len(test_keys)}) splits."
    )

    # Check reasonable split size
    total_records = len(train_df) + len(test_df)
    assert total_records == len(df), "Row counts before and after split do not match!"
    assert 0.15 <= len(test_df) / total_records <= 0.25, "Test split ratio out of expected range."


# ==============================================================================
# Test 2: Target & Derived Targets Never in Feature Sets
# ==============================================================================
def test_g3_never_in_feature_set():
    """Verify G3 and derived target columns are strictly omitted from all scenarios."""
    feature_sets = get_feature_sets()

    for scenario_name in ["scenario_a", "scenario_b"]:
        features = feature_sets[scenario_name]["all"]

        # Ensure G3 is never an input feature
        assert TARGET_COL not in features, (
            f"CRITICAL TARGET LEAKAGE: '{TARGET_COL}' found in {scenario_name}!"
        )

        # Ensure no derived target flags leak
        for leak_col in LEAKAGE_COLS:
            assert leak_col not in features, (
                f"LEAKAGE VIOLATION: '{leak_col}' found in {scenario_name}!"
            )

        # Ensure identity hash is not a feature
        assert "student_key" not in features, (
            f"LEAKAGE VIOLATION: 'student_key' found in {scenario_name}!"
        )


# ==============================================================================
# Test 3: Sensitive Attributes Strictly Excluded
# ==============================================================================
def test_sensitive_columns_excluded():
    """Verify protected demographic and socioeconomic features are never predictive features."""
    feature_sets = get_feature_sets()

    for scenario_name in ["scenario_a", "scenario_b"]:
        features = set(feature_sets[scenario_name]["all"])

        # Check all sensitive features
        for sensitive_col in SENSITIVE_FEATURES:
            assert sensitive_col not in features, (
                f"ALGORITHMIC FAIRNESS VIOLATION: Sensitive feature '{sensitive_col}' "
                f"found in {scenario_name} feature set!"
            )


# ==============================================================================
# Test 4: Unseen Categories at Transform Time Do Not Crash
# ==============================================================================
def test_unseen_category_at_transform_time(combined_raw_data: pd.DataFrame):
    """Verify OneHotEncoder(handle_unknown='ignore') handles unseen categories gracefully."""
    df = combined_raw_data.copy()
    cleaned_df, _ = clean(df)
    X, _, _, numeric_cols, categorical_cols = prepare_scenario_data(cleaned_df, scenario="A")

    # Fit preprocessor strictly on training data
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    preprocessor.fit(X)

    # Expected transformed column dimensionality
    train_transformed = preprocessor.transform(X)
    expected_dim = train_transformed.shape[1]

    # Create unseen test record with unknown categories
    unseen_row = X.iloc[[0]].copy()
    unseen_row.loc[unseen_row.index[0], "school"] = "UNKNOWN_NEW_SCHOOL"
    unseen_row.loc[unseen_row.index[0], "reason"] = "MARTIAN_EXCHANGE_PROGRAM"
    unseen_row.loc[unseen_row.index[0], "subject"] = "QUANTUM_PHYSICS"

    # Transform must not raise ValueError
    try:
        transformed_unseen = preprocessor.transform(unseen_row)
    except Exception as e:
        pytest.fail(f"Preprocessor crashed on unseen category: {e}")

    # Output dimension must be preserved (unknown categories mapped to all zeros)
    assert transformed_unseen.shape[1] == expected_dim, (
        f"Transformed dimension mismatch: expected {expected_dim}, got {transformed_unseen.shape[1]}"
    )
    assert not np.isnan(transformed_unseen).any(), "NaN found in transformed output array."


# ==============================================================================
# Test 5: Clean Function Range Repairs & Deduplication Audit
# ==============================================================================
def test_clean_validation_and_reporting():
    """Verify clean() detects, repairs, and reports anomalies (duplicates, invalid ranges, nulls)."""
    dirty_data = pd.DataFrame(
        {
            "school": ["GP", "GP", "MS"],
            "sex": ["F", "F", "M"],
            "age": [17, 17, 18],
            "address": ["U", "U", "R"],
            "famsize": ["GT3", "GT3", "LE3"],
            "Pstatus": ["T", "T", "A"],
            "Medu": [4, 4, 2],
            "Fedu": [4, 4, 2],
            "Mjob": ["teacher", "teacher", "services"],
            "Fjob": ["services", "services", "other"],
            "reason": ["home", "home", "reputation"],
            "guardian": ["mother", "mother", "father"],
            "traveltime": [2, 2, np.nan],  # Missing value
            "studytime": [2, 2, 2],
            "failures": [0, 0, 9],         # Out of range (> 4)
            "schoolsup": ["no", "no", "yes"],
            "famsup": ["yes", "yes", "no"],
            "paid": ["no", "no", "yes"],
            "activities": ["yes", "yes", "no"],
            "nursery": ["yes", "yes", "yes"],
            "higher": ["yes", "yes", "yes"],
            "internet": ["yes", "yes", "no"],
            "romantic": ["no", "no", "yes"],
            "famrel": [4, 4, 4],
            "freetime": [3, 3, 3],
            "goout": [3, 3, 3],
            "Dalc": [1, 1, 1],
            "Walc": [1, 1, 1],
            "health": [3, 3, 3],
            "absences": [4, 4, -8],        # Negative absences
            "G1": [12, 12, 10],
            "G2": [13, 13, 11],
            "G3": [14, 14, 25],            # Grade > 20
        }
    )

    cleaned_df, report = clean(dirty_data)

    # Exact duplicate was dropped
    assert report["exact_duplicates_removed"] == 1
    assert len(cleaned_df) == 2

    # Negative absences repaired
    assert (cleaned_df["absences"] >= 0).all()
    assert report["range_violations_repaired"] > 0

    # Grade clipped to <= 20
    assert (cleaned_df["G3"] <= 20).all()

    # Failures clipped to <= 4
    assert (cleaned_df["failures"] <= 4).all()

    # Missing traveltime imputed
    assert cleaned_df["traveltime"].isnull().sum() == 0
    assert report["missing_values_handled"] == 1
    assert report["status"] == "REPAIRED"


# ==============================================================================
# Test 6: Target Generation & Risk Pass Mark
# ==============================================================================
def test_make_targets():
    """Verify target derivation respects the RISK_PASS_MARK (10.0) operational definition."""
    sample_df = pd.DataFrame(
        {
            "G3": [0.0, 5.0, 9.5, 10.0, 14.0, 20.0],
        }
    )

    y_reg, y_clf = make_targets(sample_df)

    # Check regression values match G3 exactly
    np.testing.assert_array_equal(y_reg.values, [0.0, 5.0, 9.5, 10.0, 14.0, 20.0])

    # Check at-risk definition: G3 < 10.0 is 1 (at-risk), G3 >= 10.0 is 0 (pass)
    expected_clf = [1, 1, 1, 0, 0, 0]
    np.testing.assert_array_equal(y_clf.values, expected_clf)


# ==============================================================================
# Test 7: Scikit-Learn Pipeline Integration (Train-Only Fit)
# ==============================================================================
def test_pipeline_fit_only_on_train(combined_raw_data: pd.DataFrame):
    """Verify preprocessor fits inside an end-to-end sklearn Pipeline with zero leakage."""
    df = combined_raw_data.copy()
    cleaned_df, _ = clean(df)
    train_df, test_df = grouped_train_test_split(cleaned_df, test_size=0.2, random_state=RANDOM_STATE)

    X_train, y_train_reg, _, num_cols, cat_cols = prepare_scenario_data(train_df, scenario="A")
    X_test, y_test_reg, _, _, _ = prepare_scenario_data(test_df, scenario="A")

    preprocessor = build_preprocessor(num_cols, cat_cols)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", Ridge(alpha=1.0)),
        ]
    )

    # Fit STRICTLY on training split
    pipeline.fit(X_train, y_train_reg)

    # Predict on test split
    preds = pipeline.predict(X_test)
    assert len(preds) == len(X_test)
    assert not np.isnan(preds).any()

