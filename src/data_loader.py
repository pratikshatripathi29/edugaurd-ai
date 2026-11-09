"""EduGuard AI - Data Ingestion and Validation Module (Stage 1).

This module handles:
1. Loading the UCI Student Performance raw datasets (semicolon-separated).
2. Schema and value-range validation.
3. Deterministic student key generation to link the ~382 overlapping students
   between Math and Portuguese cohorts (Cortez & Silva, 2008).
4. Unification of datasets with a 'subject' indicator and an operational 'at_risk' flag.
5. Strict distinction between real data and simulated records.
"""

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from src.config import (
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    RISK_PASS_MARK,
    STUDENT_MAT_CSV,
    STUDENT_MERGE_KEYS,
    STUDENT_POR_CSV,
    TARGET_COL,
)


def generate_student_key(row: pd.Series) -> str:
    """Generate a deterministic, anonymized hash key for student identity.

    Based on the Cortez & Silva (2008) student identification attributes:
    ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
     'Mjob', 'Fjob', 'reason', 'nursery', 'internet'].
    """
    key_components = [str(row[col]).strip().lower() for col in STUDENT_MERGE_KEYS]
    raw_str = "_".join(key_components)
    return hashlib.sha256(raw_str.encode("utf-8")).hexdigest()[:12]


def validate_schema(df: pd.DataFrame, dataset_name: str = "Student Dataset") -> bool:
    """Validate that required columns exist, types are sound, and values fall in valid ranges.

    Raises:
        ValueError: If schema or value constraints are violated.
    """
    required_cols = STUDENT_MERGE_KEYS + [
        "traveltime", "studytime", "failures", "schoolsup", "famsup",
        "paid", "activities", "higher", "romantic", "famrel", "freetime",
        "goout", "Dalc", "Walc", "health", "absences", "G1", "G2", "G3"
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{dataset_name}] Missing required columns: {missing}")

    # Check grade range [0, 20]
    for grade_col in ["G1", "G2", "G3"]:
        if grade_col in df.columns:
            invalid_grades = df[(df[grade_col] < 0) | (df[grade_col] > 20)]
            if not invalid_grades.empty:
                raise ValueError(
                    f"[{dataset_name}] Found {len(invalid_grades)} rows with {grade_col} outside [0, 20]."
                )

    # Check non-negative absences
    if (df["absences"] < 0).any():
        raise ValueError(f"[{dataset_name}] Found negative values in 'absences'.")

    return True


def load_raw_dataset(subject: str = "mat") -> pd.DataFrame:
    """Load a single raw dataset ('mat' or 'por') from the raw data directory.

    Args:
        subject: 'mat' for Mathematics, 'por' for Portuguese.

    Returns:
        pd.DataFrame with parsed columns.
    """
    subject = subject.lower().strip()
    if subject in ["mat", "math", "mathematics"]:
        path = STUDENT_MAT_CSV
        name = "Math"
    elif subject in ["por", "portuguese"]:
        path = STUDENT_POR_CSV
        name = "Portuguese"
    else:
        raise ValueError(f"Unknown subject '{subject}'. Expected 'mat' or 'por'.")

    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found at: {path}")

    df = pd.read_csv(path, sep=";")
    validate_schema(df, dataset_name=name)
    return df


def load_and_combine_data(
    save_processed: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load both Math and Portuguese datasets, generate student keys, and build unified view.

    Returns:
        Tuple containing:
        - df_mat: Mathematics DataFrame with student_key, subject, at_risk
        - df_por: Portuguese DataFrame with student_key, subject, at_risk
        - df_combined: Stacked DataFrame containing all student enrollments (1044 records)
    """
    df_mat = load_raw_dataset("mat").copy()
    df_por = load_raw_dataset("por").copy()

    # Assign metadata
    df_mat["subject"] = "math"
    df_por["subject"] = "portuguese"

    # Flag simulation status (Real UCI data is NEVER simulated)
    df_mat["is_simulated"] = False
    df_por["is_simulated"] = False

    # Deterministic student identity key across both datasets
    df_mat["student_key"] = df_mat.apply(generate_student_key, axis=1)
    df_por["student_key"] = df_por.apply(generate_student_key, axis=1)

    # Operational binary risk definition: G3 < RISK_PASS_MARK (10.0)
    df_mat["at_risk"] = (df_mat[TARGET_COL] < RISK_PASS_MARK).astype(int)
    df_por["at_risk"] = (df_por[TARGET_COL] < RISK_PASS_MARK).astype(int)

    # Combined stacked dataset
    df_combined = pd.concat([df_mat, df_por], ignore_index=True)

    if save_processed:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        df_mat.to_csv(PROCESSED_DATA_DIR / "student_mat_clean.csv", index=False)
        df_por.to_csv(PROCESSED_DATA_DIR / "student_por_clean.csv", index=False)
        df_combined.to_csv(PROCESSED_DATA_DIR / "student_combined.csv", index=False)

    return df_mat, df_por, df_combined


def get_shared_student_keys() -> List[str]:
    """Return the list of student keys representing students enrolled in both subjects."""
    df_mat = load_raw_dataset("mat")
    df_por = load_raw_dataset("por")

    keys_mat = set(df_mat.apply(generate_student_key, axis=1))
    keys_por = set(df_por.apply(generate_student_key, axis=1))

    shared = list(keys_mat.intersection(keys_por))
    return shared
