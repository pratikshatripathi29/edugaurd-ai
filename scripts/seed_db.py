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

from src.classification import load_classification_model, predict_risk
from src.config import DATABASE_PATH, PROCESSED_DATA_DIR
from src.database import (
    AcademicRecord,
    Alert,
    Intervention,
    Prediction,
    Student,
    WeeklyPerformance,
    get_db_session,
    init_db,
    reset_db,
)
from src.monitoring import monitor_cohort_weekly_transition
from src.regression import load_model, predict_score
from src.simulation import generate_simulated_weekly_dataset


def seed_database(reset: bool = True) -> Dict[str, int]:
    """Execute full database seeding pipeline.

    Args:
        reset: If True, drops and recreates all tables before seeding.

    Returns:
        Dictionary reporting total row counts per table.
    """
    print(f"Connecting to database at: {DATABASE_PATH}")
    if reset:
        print("Resetting database schema...")
        reset_db()
    else:
        init_db()

    test_csv_path = PROCESSED_DATA_DIR / "test.csv"
    if not test_csv_path.exists():
        raise FileNotFoundError(f"Required test split not found at: {test_csv_path}")

    print(f"Loading real test students from {test_csv_path}...")
    df_test = pd.read_csv(test_csv_path)
    n_students = len(df_test)

    # 1. Assign sequential IDs
    df_test["student_id"] = [f"EDU-{i+1:04d}" for i in range(n_students)]

    # 2. Insert Students & AcademicRecords
    print(f"Inserting {n_students} Students and AcademicRecords...")
    student_records = []
    academic_records = []

    for _, row in df_test.iterrows():
        st_id = row["student_id"]

        st_obj = Student(
            student_id=st_id,
            student_key=str(row["student_key"]) if pd.notna(row.get("student_key")) else None,
            school=str(row["school"]) if pd.notna(row.get("school")) else None,
            sex=str(row["sex"]) if pd.notna(row.get("sex")) else None,
            age=int(row["age"]) if pd.notna(row.get("age")) else None,
            address=str(row["address"]) if pd.notna(row.get("address")) else None,
            famsize=str(row["famsize"]) if pd.notna(row.get("famsize")) else None,
            Pstatus=str(row["Pstatus"]) if pd.notna(row.get("Pstatus")) else None,
            Medu=int(row["Medu"]) if pd.notna(row.get("Medu")) else None,
            Fedu=int(row["Fedu"]) if pd.notna(row.get("Fedu")) else None,
            Mjob=str(row["Mjob"]) if pd.notna(row.get("Mjob")) else None,
            Fjob=str(row["Fjob"]) if pd.notna(row.get("Fjob")) else None,
            reason=str(row["reason"]) if pd.notna(row.get("reason")) else None,
            guardian=str(row["guardian"]) if pd.notna(row.get("guardian")) else None,
            data_source="REAL_UCI",
        )
        student_records.append(st_obj)

        acad_obj = AcademicRecord(
            student_id=st_id,
            subject=str(row.get("subject", "math")),
            traveltime=int(row["traveltime"]) if pd.notna(row.get("traveltime")) else None,
            studytime=int(row["studytime"]) if pd.notna(row.get("studytime")) else None,
            failures=int(row["failures"]) if pd.notna(row.get("failures")) else None,
            schoolsup=str(row["schoolsup"]) if pd.notna(row.get("schoolsup")) else None,
            famsup=str(row["famsup"]) if pd.notna(row.get("famsup")) else None,
            paid=str(row["paid"]) if pd.notna(row.get("paid")) else None,
            activities=str(row["activities"]) if pd.notna(row.get("activities")) else None,
            nursery=str(row["nursery"]) if pd.notna(row.get("nursery")) else None,
            higher=str(row["higher"]) if pd.notna(row.get("higher")) else None,
            internet=str(row["internet"]) if pd.notna(row.get("internet")) else None,
            romantic=str(row["romantic"]) if pd.notna(row.get("romantic")) else None,
            famrel=int(row["famrel"]) if pd.notna(row.get("famrel")) else None,
            freetime=int(row["freetime"]) if pd.notna(row.get("freetime")) else None,
            goout=int(row["goout"]) if pd.notna(row.get("goout")) else None,
            Dalc=int(row["Dalc"]) if pd.notna(row.get("Dalc")) else None,
            Walc=int(row["Walc"]) if pd.notna(row.get("Walc")) else None,
            health=int(row["health"]) if pd.notna(row.get("health")) else None,
            absences=int(row["absences"]) if pd.notna(row.get("absences")) else None,
            G1=float(row["G1"]) if pd.notna(row.get("G1")) else None,
            G2=float(row["G2"]) if pd.notna(row.get("G2")) else None,
            G3=float(row["G3"]) if pd.notna(row.get("G3")) else None,
            data_source="REAL_UCI",
        )
        academic_records.append(acad_obj)

    with get_db_session() as session:
        session.add_all(student_records)
        session.add_all(academic_records)

    # 3. Insert Simulated Weekly Performance (Weeks 1-12)
    print("Generating and inserting SIMULATED weekly records...")
    sim_path = PROCESSED_DATA_DIR / "simulated_weekly_performance.csv"
    if sim_path.exists():
        df_sim = pd.read_csv(sim_path)
    else:
        df_sim = generate_simulated_weekly_dataset(df_test, output_path=sim_path)

    weekly_objs = []
    for _, row in df_sim.iterrows():
        w_obj = WeeklyPerformance(
            student_id=str(row["student_id"]),
            week_number=int(row["week_number"]),
            attendance_rate=float(row["attendance_rate"]),
            quiz_average=float(row["quiz_average"]),
            assignment_completion=float(row["assignment_completion"]),
            study_hours=float(row["study_hours"]),
            trajectory_type=str(row.get("trajectory_type", "STABLE")),
            notes=str(row.get("notes", "SIMULATED")),
            data_source="SIMULATED",
        )
        weekly_objs.append(w_obj)

    with get_db_session() as session:
        session.add_all(weekly_objs)

    # 4. Generate Predictions from Trained Models
    print("Running final models to populate Predictions table...")
    # Prepare clean input without target columns
    drop_cols = ["G3", "at_risk", "student_key", "is_simulated", "student_id"]
    candidate_df = df_test.drop(columns=[c for c in drop_cols if c in df_test.columns], errors="ignore")

    # Load models
    clf_a, meta_clf_a = load_classification_model("A")
    reg_a, meta_reg_a = load_model("A")
    clf_b, meta_clf_b = load_classification_model("B")
    reg_b, meta_reg_b = load_model("B")

    # Inference Scenario A
    pred_clf_a = predict_risk(clf_a, meta_clf_a, candidate_df, threshold=0.25)
    pred_reg_a = predict_score(reg_a, meta_reg_a, candidate_df)

    # Inference Scenario B
    pred_clf_b = predict_risk(clf_b, meta_clf_b, candidate_df, threshold=0.15)
    pred_reg_b = predict_score(reg_b, meta_reg_b, candidate_df)

    pred_records = []
    for i in range(n_students):
        st_id = df_test.iloc[i]["student_id"]

        # Scenario A record
        p_a = Prediction(
            student_id=st_id,
            scenario="Scenario A",
            model_version=meta_clf_a.get("model_version", "v1.0-early"),
            predicted_score=float(pred_reg_a[i]),
            at_risk_probability=float(pred_clf_a["probabilities"][i]),
            is_at_risk=int(pred_clf_a["predictions"][i]),
            risk_category=str(pred_clf_a["risk_tiers"][i]),
            top_risk_factors=json.dumps(["failures", "absences", "studytime"]),
            data_source="REAL_UCI",
        )
        pred_records.append(p_a)

        # Scenario B record
        p_b = Prediction(
            student_id=st_id,
            scenario="Scenario B",
            model_version=meta_clf_b.get("model_version", "v1.0-mid"),
            predicted_score=float(pred_reg_b[i]),
            at_risk_probability=float(pred_clf_b["probabilities"][i]),
            is_at_risk=int(pred_clf_b["predictions"][i]),
            risk_category=str(pred_clf_b["risk_tiers"][i]),
            top_risk_factors=json.dumps(["G2", "G1", "grade_trend"]),
            data_source="REAL_UCI",
        )
        pred_records.append(p_b)

    with get_db_session() as session:
        session.add_all(pred_records)

    # 5. Run Monitoring to Populate Alerts
    print("Evaluating weekly transitions to generate Alerts...")
    with get_db_session() as session:
        monitor_cohort_weekly_transition(df_sim, session=session)

    # 6. Create Sample Interventions for High-Risk Alerts
    print("Creating sample Interventions for triage demonstration...")
    with get_db_session() as session:
        high_alerts = session.scalars(select(Alert).where(Alert.risk_level == "High")).all()
        interventions = []
        for alert in high_alerts[:12]:
            interv = Intervention(
                student_id=alert.student_id,
                alert_id=alert.id,
                intervention_type="MANDATORY_TUTORING",
                description=f"Assigned 2 hours/week remedial study group following alert: {alert.reason}",
                assigned_to="Academic Success Center",
                status="Scheduled",
                data_source="SIMULATED",
            )
            interventions.append(interv)
        session.add_all(interventions)

    # 7. Print Table Summary
    summary = {}
    with get_db_session() as session:
        summary["Students"] = session.scalar(select(func.count(Student.id))) or 0
        summary["AcademicRecords"] = session.scalar(select(func.count(AcademicRecord.id))) or 0
        summary["WeeklyPerformance"] = session.scalar(select(func.count(WeeklyPerformance.id))) or 0
        summary["Predictions"] = session.scalar(select(func.count(Prediction.id))) or 0
        summary["Alerts"] = session.scalar(select(func.count(Alert.id))) or 0
        summary["Interventions"] = session.scalar(select(func.count(Intervention.id))) or 0

    print("\n" + "=" * 50)
    print("DATABASE SEEDING COMPLETED SUCCESSFULLY")
    print("=" * 50)
    for table, count in summary.items():
        print(f"  • {table:<22}: {count:>6} rows")
    print("=" * 50 + "\n")

    return summary


if __name__ == "__main__":
    seed_database(reset=True)

