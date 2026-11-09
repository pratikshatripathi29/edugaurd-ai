"""EduGuard AI - Preprocessing and Grouped Splitting Module (Stage 2).

This module guarantees strict zero-data-leakage preprocessing:
1. Data cleaning and range validation with a transparent audit report.
2. Deterministic student key generation for cohort linking.
3. Grouped train/test splitting (GroupShuffleSplit) preventing student identity leakage.
4. Construction of scikit-learn ColumnTransformer objects designed to be fitted
   strictly on training data within full Pipeline objects.
"""

import hashlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    RANDOM_STATE,
    STUDENT_MERGE_KEYS,
)


def make_student_key(df: pd.DataFrame) -> pd.Series:
    """Generate a deterministic, anonymized hash key for student identity.

    Based on the Cortez & Silva (2008) student identification attributes:
    ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
     'Mjob', 'Fjob', 'reason', 'nursery', 'internet'].

    Args:
        df: Input DataFrame containing the identification columns.

    Returns:
        pd.Series of 12-character SHA-256 hex strings.
    """
    for col in STUDENT_MERGE_KEYS:
        if col not in df.columns:
            raise KeyError(f"Missing required student identification key column: '{col}'")

    def _hash_row(row: pd.Series) -> str:
        key_parts = [str(row[c]).strip().lower() for c in STUDENT_MERGE_KEYS]
        raw_key = "_".join(key_parts)
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:12]

    return df.apply(_hash_row, axis=1)


def clean(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Validate ranges, repair invalid values, drop exact duplicates, and return audit report.

    Rules enforced:
    - Grades (G1, G2, G3) must lie within [0.0, 20.0]. Values outside are clipped or flagged.
    - Absences must be non-negative (clipped at 0).
    - Ratings (traveltime, studytime, famrel, freetime, goout, Dalc, Walc, health) in [1, 5].
    - Failures in [0, 4].
    - Exact duplicate rows are dropped.
    - Missing values (if any) are reported and median/mode imputed for numeric/categorical columns.

    Args:
        df: Input raw or concatenated DataFrame.

    Returns:
        Tuple containing:
        - cleaned_df: Sanitized, validated DataFrame copy.
        - report: Dictionary detailing row counts, duplicates removed, and repairs made.
    """
    cleaned = df.copy()
    initial_rows = len(cleaned)
    actions: List[str] = []

    # 1. Deduplication
    dup_mask = cleaned.duplicated()
    exact_duplicates_removed = int(dup_mask.sum())
    if exact_duplicates_removed > 0:
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)
        actions.append(f"Removed {exact_duplicates_removed} exact duplicate rows.")

    # 2. Missing value detection & handling
    missing_counts = cleaned.isnull().sum()
    total_missing = int(missing_counts.sum())
    if total_missing > 0:
        for col in cleaned.columns:
            n_miss = int(cleaned[col].isnull().sum())
            if n_miss > 0:
                if pd.api.types.is_numeric_dtype(cleaned[col]):
                    median_val = cleaned[col].median()
                    cleaned[col] = cleaned[col].fillna(median_val)
                    actions.append(f"Imputed {n_miss} missing values in '{col}' with median ({median_val}).")
                else:
                    mode_val = cleaned[col].mode()[0]
                    cleaned[col] = cleaned[col].fillna(mode_val)
                    actions.append(f"Imputed {n_miss} missing values in '{col}' with mode ('{mode_val}').")

    # 3. Value Range Validation & Repair
    range_repairs = 0

    # Grades validation: [0, 20]
    for grade_col in ["G1", "G2", "G3"]:
        if grade_col in cleaned.columns:
            invalid_low = (cleaned[grade_col] < 0)
            invalid_high = (cleaned[grade_col] > 20)
            n_invalid = int((invalid_low | invalid_high).sum())
            if n_invalid > 0:
                cleaned[grade_col] = cleaned[grade_col].clip(lower=0.0, upper=20.0)
                range_repairs += n_invalid
                actions.append(f"Clipped {n_invalid} out-of-range values in '{grade_col}' to [0, 20].")

    # Absences validation: >= 0
    if "absences" in cleaned.columns:
        neg_absences = int((cleaned["absences"] < 0).sum())
        if neg_absences > 0:
            cleaned["absences"] = cleaned["absences"].clip(lower=0)
            range_repairs += neg_absences
            actions.append(f"Clipped {neg_absences} negative absences to 0.")

    # Scale 1-5 validation
    rating_cols = ["traveltime", "studytime", "famrel", "freetime", "goout", "Dalc", "Walc", "health"]
    for col in rating_cols:
        if col in cleaned.columns:
            out_of_bounds = (cleaned[col] < 1) | (cleaned[col] > 5)
            n_oob = int(out_of_bounds.sum())
            if n_oob > 0:
                cleaned[col] = cleaned[col].clip(lower=1, upper=5)
                range_repairs += n_oob
                actions.append(f"Clipped {n_oob} ratings in '{col}' to [1, 5].")

    # Failures: [0, 4]
    if "failures" in cleaned.columns:
        invalid_failures = (cleaned["failures"] < 0) | (cleaned["failures"] > 4)
        n_fail = int(invalid_failures.sum())
        if n_fail > 0:
            cleaned["failures"] = cleaned["failures"].clip(lower=0, upper=4)
            range_repairs += n_fail
            actions.append(f"Clipped {n_fail} invalid values in 'failures' to [0, 4].")

    # 4. Generate report
    report: Dict[str, Any] = {
        "initial_rows": initial_rows,
        "final_rows": len(cleaned),
        "exact_duplicates_removed": exact_duplicates_removed,
        "missing_values_handled": total_missing,
        "range_violations_repaired": range_repairs,
        "actions_taken": actions,
        "status": "CLEAN" if (exact_duplicates_removed == 0 and total_missing == 0 and range_repairs == 0) else "REPAIRED",
    }

    return cleaned, report


def grouped_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
    group_col: str = "student_key",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Perform leak-free train/test split grouped by student identity key.

    Uses GroupShuffleSplit to guarantee that all records belonging to the same
    underlying student (who may be enrolled in both Math and Portuguese) fall
    strictly into EITHER the training set OR the test set.

    Args:
        df: Input DataFrame.
        test_size: Fraction of student groups to assign to the test set (default 0.2).
        random_state: Seed for reproducibility.
        group_col: Column name containing student identifiers (generated if missing).

    Returns:
        Tuple of (train_df, test_df) DataFrames.

    Raises:
        AssertionError: If any student_key appears in both train and test splits.
    """
    df_split = df.copy()
    if group_col not in df_split.columns:
        df_split[group_col] = make_student_key(df_split)

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(df_split, groups=df_split[group_col]))

    train_df = df_split.iloc[train_idx].copy().reset_index(drop=True)
    test_df = df_split.iloc[test_idx].copy().reset_index(drop=True)

    # CRITICAL LEAKAGE ASSERTION: Zero shared students across splits
    overlap = set(train_df[group_col]).intersection(set(test_df[group_col]))
    assert len(overlap) == 0, (
        f"CRITICAL DATA LEAKAGE: {len(overlap)} student_key values found in both "
        f"train and test sets! Split violated group isolation."
    )

    return train_df, test_df


def build_preprocessor(
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> ColumnTransformer:
    """Build a scikit-learn ColumnTransformer for numerical and categorical feature pipelines.

    Architecture:
    - Numeric Pipeline:
        1. SimpleImputer(strategy='median')
        2. StandardScaler()
    - Categorical Pipeline:
        1. SimpleImputer(strategy='most_frequent')
        2. OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    NOTE ON LEAKAGE PREVENTION:
        This ColumnTransformer must be integrated inside an `sklearn.pipeline.Pipeline`
        or fitted strictly via `preprocessor.fit(X_train)` before calling
        `preprocessor.transform(X_test)`. Never call fit or fit_transform on the whole dataset.

    Args:
        numeric_cols: List of numerical feature column names.
        categorical_cols: List of categorical feature column names.

    Returns:
        Configured ColumnTransformer instance.
    """
    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numeric_cols),
            ("cat", cat_pipeline, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor
