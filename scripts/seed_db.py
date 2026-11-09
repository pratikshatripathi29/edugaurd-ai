"""
EduGuard AI — Database Seeding Script (Stage 6)

Seeds the production SQLite database with:
1. Real UCI held-out test students (data/processed/test.csv) -> Students & AcademicRecords
2. Synthetic longitudinal weekly logs (Weeks 1-12) -> WeeklyPerformance (data_source = 'SIMULATED')
3. Model inference predictions (Scenario A & Scenario B) -> Predictions
4. Longitudinal early-warning monitoring alerts -> Alerts
5. Advisory counselor intervention records -> Interventions
"""

import json
from pathlib import Path
import sys
from typing import Any, Dict, List

# Ensure project root is in sys.path when invoked directly
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
from sqlalchemy import func, select

from src.config import DATABASE_PATH
from src.database import seed_database_from_processed


def seed_database(reset: bool = True) -> Dict[str, int]:
    """Execute full database seeding pipeline using committed processed datasets and models.

    Args:
        reset: If True, drops and recreates all tables before seeding.

    Returns:
        Dictionary reporting total row counts per table.
    """
    print(f"Connecting to database at: {DATABASE_PATH}")
    summary = seed_database_from_processed(reset=reset)

    print("\n" + "=" * 50)
    print("DATABASE SEEDING COMPLETED SUCCESSFULLY")
    print("=" * 50)
    for table, count in summary.items():
        print(f"  • {table:<22}: {count:>6} rows")
    print("=" * 50 + "\n")

    return summary


if __name__ == "__main__":
    seed_database(reset=True)

