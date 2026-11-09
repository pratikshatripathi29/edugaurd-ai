"""EduGuard AI - Central Configuration Module.

This module defines directory paths, reproducibility seeds, operational risk definitions,
thresholds, and strict feature governance boundaries.

CRITICAL ETHICAL & MODELING GOVERNANCE:
    Features in SENSITIVE_FEATURES must NEVER be passed as training or inference inputs
    into predictive models. They are reserved strictly for post-hoc fairness audits
    and demographic parity / equal opportunity evaluations.
"""

from pathlib import Path
from typing import Dict, List

# ==============================================================================
# 1. Project Directory Structure (Pathlib)
# ==============================================================================
# Root directory is the parent of the `src` folder (eduguard-ai/)
ROOT_DIR: Path = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR: Path = ROOT_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

# Source code & application directories
SRC_DIR: Path = ROOT_DIR / "src"
PAGES_DIR: Path = ROOT_DIR / "pages"
ASSETS_DIR: Path = ROOT_DIR / "assets"
NOTEBOOKS_DIR: Path = ROOT_DIR / "notebooks"
MODELS_DIR: Path = ROOT_DIR / "models"
REPORTS_DIR: Path = ROOT_DIR / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
TESTS_DIR: Path = ROOT_DIR / "tests"

# Known raw data file paths (UCI Student Performance dataset)
STUDENT_MAT_CSV: Path = RAW_DATA_DIR / "student-mat.csv"
STUDENT_POR_CSV: Path = RAW_DATA_DIR / "student-por.csv"

# SQLite Database path for monitoring and audit logs
DATABASE_PATH: Path = ROOT_DIR / "eduguard.db"
SQL_DATABASE_URL: str = f"sqlite:///{DATABASE_PATH}"

# Ensure required runtime output directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# 2. Reproducibility & Random State
# ==============================================================================
RANDOM_STATE: int = 42


# ==============================================================================
# 3. Operational Risk Definitions & Thresholds
# ==============================================================================
# In the Portuguese educational grading system (0 - 20 scale):
# G3 < 10 represents a failing grade.
# NOTE: This is an OPERATIONAL definition for early-warning support, not an intrinsic label.
RISK_PASS_MARK: float = 10.0

# Calibrated classification probability cut-offs for risk categorization:
# - Low Risk:       P(At-Risk) < 0.30
# - Moderate Risk:  0.30 <= P(At-Risk) <= 0.60
# - High Risk:      P(At-Risk) > 0.60
RISK_THRESHOLDS: Dict[str, float] = {
    "low": 0.30,
    "high": 0.60,
}


# ==============================================================================
# 4. Feature Governance & Leakage Prevention (STRICT RULES)
# ==============================================================================
# Target variable in the UCI Student Performance dataset
TARGET_COL: str = "G3"

# Period 1 and Period 2 interim grades
INTERIM_GRADE_COLS: List[str] = ["G1", "G2"]

# Target leakage columns: Attributes that directly reveal or leak the target
LEAKAGE_COLS: List[str] = [
    "G3",        # G3 must NEVER be an input to predict G3
    "passed",    # Derived target indicators
    "at_risk",   # Derived binary target indicator
]

# Sensitive / Protected Attributes:
# These features reflect demographic, familial, or socio-economic identities.
# RULE: MUST NEVER BE FED AS INPUT FEATURES TO PREDICTIVE MODELS.
# They are only retained for post-hoc fairness audits (disparate impact, equalized odds).
SENSITIVE_FEATURES: List[str] = [
    "sex",          # Student sex ('F' - female, 'M' - male)
    "age",          # Student age (numeric: 15 to 22)
    "address",      # Student home address type ('U' - urban, 'R' - rural)
    "famsize",      # Family size ('LE3' - <= 3, 'GT3' - > 3)
    "Pstatus",      # Parent's cohabitation status ('T' - living together, 'A' - apart)
    "Medu",         # Mother's education (0 - none, 1 - 4th grade, 2 - 5th-9th, 3 - secondary, 4 - higher)
    "Fedu",         # Father's education (0 - none, 1 - 4th grade, 2 - 5th-9th, 3 - secondary, 4 - higher)
    "Mjob",         # Mother's job ('teacher', 'health' care, civil 'services', 'at_home', 'other')
    "Fjob",         # Father's job ('teacher', 'health' care, civil 'services', 'at_home', 'other')
    "guardian",     # Student's guardian ('mother', 'father', 'other')
    "nationality",  # Nationality or citizenship fields (if present in extensions/simulations)
]

# Columns used to uniquely identify duplicate students who take both Math and Portuguese:
# Cortez and Silva (2008) identified that 382 students appear in both student-mat.csv and student-por.csv.
STUDENT_MERGE_KEYS: List[str] = [
    "school",
    "sex",
    "age",
    "address",
    "famsize",
    "Pstatus",
    "Medu",
    "Fedu",
    "Mjob",
    "Fjob",
    "reason",
    "nursery",
    "internet",
]


# ==============================================================================
# 5. Modeling Scenarios
# ==============================================================================
# Scenario A (Early-Semester / Early Warning):
# Evaluates student trajectory BEFORE first/second period exam grades exist.
# interim grade columns (G1, G2) are strictly excluded.
#
# Scenario B (Mid-Semester Update):
# Evaluates student trajectory AFTER G1 and G2 have been recorded.
# interim grade columns (G1, G2) are included as predictive features.
SCENARIO_A: str = "Scenario A (Early-Warning, No G1/G2)"
SCENARIO_B: str = "Scenario B (Mid-Semester, With G1/G2)"

