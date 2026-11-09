"""
Unit Tests for EduGuard AI Data Loading, Parsing & Schema Validation (Stage 11).

Validates:
1. Semicolon-delimited parsing of UCI raw CSVs (student-mat.csv and student-por.csv).
2. Schema validation enforcing required columns, absence bounds, and [0, 20] mark ranges.
3. Deterministic student key generation for the ~382 overlapping students.
4. Exception handling for unknown subjects, invalid marks, and missing fields.
"""

import pandas as pd
import pytest

from src.data_loader import (
    generate_student_key,
    get_shared_student_keys,
    load_and_combine_data,
    load_raw_dataset,
    validate_schema,
)


def test_load_raw_math_dataset():
    """Verify loading raw Math dataset correctly parses semicolon-separated values."""
    df_mat = load_raw_dataset("mat")
    assert isinstance(df_mat, pd.DataFrame)
    assert len(df_mat) == 395, f"Expected 395 math records, got {len(df_mat)}"
    assert "G3" in df_mat.columns
    assert "school" in df_mat.columns
    assert df_mat["G3"].min() >= 0.0
    assert df_mat["G3"].max() <= 20.0


def test_load_raw_portuguese_dataset():
    """Verify loading raw Portuguese dataset correctly parses 649 rows."""
    df_por = load_raw_dataset("por")
    assert isinstance(df_por, pd.DataFrame)
    assert len(df_por) == 649, f"Expected 649 Portuguese records, got {len(df_por)}"
    assert "G3" in df_por.columns


def test_load_raw_unknown_subject_raises():
    """Verify loading an invalid subject identifier raises ValueError."""
    with pytest.raises(ValueError, match="Unknown subject"):
        load_raw_dataset("chemistry")


def test_validate_schema_missing_column_raises():
    """Verify validate_schema catches missing columns."""
    df_mat = load_raw_dataset("mat").copy()
    df_incomplete = df_mat.drop(columns=["failures"])
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_schema(df_incomplete, "Incomplete Dataset")


def test_validate_schema_invalid_marks_raises():
    """Verify validate_schema detects marks outside the valid 0-20 scale."""
    df_mat = load_raw_dataset("mat").copy()
    df_mat.iloc[0, df_mat.columns.get_loc("G1")] = 25.0  # Invalid mark > 20

    with pytest.raises(ValueError, match="outside \\[0, 20\\]"):
        validate_schema(df_mat, "Invalid Mark Test")

    df_mat.iloc[0, df_mat.columns.get_loc("G1")] = -5.0  # Invalid mark < 0
    with pytest.raises(ValueError, match="outside \\[0, 20\\]"):
        validate_schema(df_mat, "Invalid Mark Test")


def test_generate_student_key_determinism():
    """Verify student key hashing is deterministic and case-insensitive."""
    row1 = pd.Series({
        "school": "GP", "sex": "F", "age": 16, "address": "U", "famsize": "GT3",
        "Pstatus": "T", "Medu": 4, "Fedu": 3, "Mjob": "health", "Fjob": "services",
        "reason": "course", "nursery": "yes", "internet": "yes",
    })
    row2 = pd.Series({
        "school": "gp", "sex": "f", "age": 16, "address": "u", "famsize": "gt3",
        "Pstatus": "t", "Medu": 4, "Fedu": 3, "Mjob": "health", "Fjob": "services",
        "reason": "course", "nursery": "yes", "internet": "yes",
    })

    key1 = generate_student_key(row1)
    key2 = generate_student_key(row2)

    assert key1 == key2, "Hash must be identical regardless of case"
    assert len(key1) == 12, "Key should be a 12-char SHA-256 slice"


def test_shared_student_keys_overlap():
    """Verify that approximately 382 students appear in both Math and Portuguese datasets."""
    shared = get_shared_student_keys()
    assert isinstance(shared, list)
    # Cortez & Silva (2008) noted ~382 duplicate students
    assert 350 <= len(shared) <= 400, f"Expected ~382 shared students, found {len(shared)}"


def test_load_and_combine_data():
    """Verify load_and_combine_data returns unified 1044 rows and flags is_simulated=False."""
    df_mat, df_por, df_comb = load_and_combine_data(save_processed=False)
    assert len(df_comb) == 395 + 649 == 1044
    assert (df_comb["is_simulated"] == False).all(), "Real UCI data must have is_simulated=False"
    assert set(df_comb["subject"]) == {"math", "portuguese"}
    assert "at_risk" in df_comb.columns
