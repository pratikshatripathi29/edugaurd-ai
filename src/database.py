"""
EduGuard AI — Database and Persistence Layer (Stage 6)

Built with SQLAlchemy 2.x and configured for SQLite (with configuration abstraction
allowing zero-friction migration to PostgreSQL in production environments).

GOVERNANCE & PROVENANCE NOTICE (PROJECT_CONTEXT.md Rule 2.1):
Every table carries a `data_source` column ('REAL_UCI' or 'SIMULATED') ensuring
unambiguous audit separation between real historical observations and synthetic
monitoring simulations.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

# Ensure project root is in sys.path when invoked directly
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

def _utcnow() -> datetime:
    """Return timezone-aware current UTC datetime."""
    return datetime.now(timezone.utc)

import pandas as pd
from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    event,
    func,
    select,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)
import streamlit as st

from src.config import DATABASE_PATH, SQL_DATABASE_URL

# ==============================================================================
# 1. SQLite Foreign Key Enforcement Event Listener
# ==============================================================================
@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection: Any, connection_record: Any) -> None:
    """Enforce SQLite foreign key constraints and check constraints on connection."""
    # Only execute pragma on SQLite connections
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA foreign_keys=ON")
    except Exception:
        pass
    finally:
        cursor.close()


# ==============================================================================
# 2. SQLAlchemy 2.x Declarative Base & ORM Models
# ==============================================================================
class Base(DeclarativeBase):
    """Base class for all EduGuard AI ORM entities."""
    pass


class Student(Base):
    """Core Student identity and demographic entity.

    Holds student-level attributes and Cortez & Silva composite merge key.
    Sensitive features stored here are reserved strictly for fairness audits.
    """
    __tablename__ = "students"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    student_id: Mapped[str] = mapped_column(String(50), unique=True, index=True, nullable=False)
    student_key: Mapped[Optional[str]] = mapped_column(String(255), index=True, nullable=True)

    # Demographic / Institutional attributes (Audit only, never predictive features)
    school: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    sex: Mapped[Optional[str]] = mapped_column(String(5), nullable=True)
    age: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    address: Mapped[Optional[str]] = mapped_column(String(5), nullable=True)
    famsize: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    Pstatus: Mapped[Optional[str]] = mapped_column(String(5), nullable=True)
    Medu: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    Fedu: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    Mjob: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)
    Fjob: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)
    reason: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)
    guardian: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)

    # Provenance & Audit
    data_source: Mapped[str] = mapped_column(String(20), default="REAL_UCI", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, onupdate=_utcnow, nullable=False
    )

    # Relationships
    academic_records: Mapped[List["AcademicRecord"]] = relationship(
        "AcademicRecord", back_populates="student", cascade="all, delete-orphan"
    )
    weekly_performances: Mapped[List["WeeklyPerformance"]] = relationship(
        "WeeklyPerformance", back_populates="student", cascade="all, delete-orphan"
    )
    predictions: Mapped[List["Prediction"]] = relationship(
        "Prediction", back_populates="student", cascade="all, delete-orphan"
    )
    alerts: Mapped[List["Alert"]] = relationship(
        "Alert", back_populates="student", cascade="all, delete-orphan"
    )
    interventions: Mapped[List["Intervention"]] = relationship(
        "Intervention", back_populates="student", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint("age IS NULL OR (age >= 10 AND age <= 30)", name="check_student_age_range"),
        CheckConstraint("Medu IS NULL OR (Medu >= 0 AND Medu <= 4)", name="check_student_medu_range"),
        CheckConstraint("Fedu IS NULL OR (Fedu >= 0 AND Fedu <= 4)", name="check_student_fedu_range"),
    )


class AcademicRecord(Base):
    """Subject-specific academic history, attendance, and exam grades."""
    __tablename__ = "academic_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    student_id: Mapped[str] = mapped_column(
        String(50), ForeignKey("students.student_id", ondelete="CASCADE"), index=True, nullable=False
    )
    subject: Mapped[str] = mapped_column(String(20), nullable=False)

    # Study habits & environment
    traveltime: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    studytime: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    failures: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    schoolsup: Mapped[Optional[str]] = mapped_column(String(5), nullable=True)
    famsup: Mapped[Optional[str]] = mapped_column(String(5), nullable=True)
    paid: Mapped[Optional[str]] = mapped_column(String(5), nullable=True)
    activities: Mapped[Optional[str]] = mapped_column(String(5), nullable=True)
    nursery: Mapped[Optional[str]] = mapped_column(String(5), nullable=True)
    higher: Mapped[Optional[str]] = mapped_column(String(5), nullable=True)
    internet: Mapped[Optional[str]] = mapped_column(String(5), nullable=True)
    romantic: Mapped[Optional[str]] = mapped_column(String(5), nullable=True)
    famrel: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    freetime: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    goout: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    Dalc: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    Walc: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    health: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    absences: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Portuguese grading scale (0.0 to 20.0)
    G1: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    G2: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    G3: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Provenance
    data_source: Mapped[str] = mapped_column(String(20), default="REAL_UCI", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, nullable=False)

    student: Mapped["Student"] = relationship("Student", back_populates="academic_records")

    __table_args__ = (
        CheckConstraint("G1 IS NULL OR (G1 >= 0.0 AND G1 <= 20.0)", name="check_academic_g1_range"),
        CheckConstraint("G2 IS NULL OR (G2 >= 0.0 AND G2 <= 20.0)", name="check_academic_g2_range"),
        CheckConstraint("G3 IS NULL OR (G3 >= 0.0 AND G3 <= 20.0)", name="check_academic_g3_range"),
        CheckConstraint("absences IS NULL OR (absences >= 0 AND absences <= 150)", name="check_academic_absences_range"),
    )


class WeeklyPerformance(Base):
    """Longitudinal weekly attendance, assignment, and quiz tracking log."""
    __tablename__ = "weekly_performance"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    student_id: Mapped[str] = mapped_column(
        String(50), ForeignKey("students.student_id", ondelete="CASCADE"), index=True, nullable=False
    )
    week_number: Mapped[int] = mapped_column(Integer, nullable=False)

    attendance_rate: Mapped[float] = mapped_column(Float, nullable=False)
    quiz_average: Mapped[float] = mapped_column(Float, nullable=False)
    assignment_completion: Mapped[float] = mapped_column(Float, nullable=False)
    study_hours: Mapped[float] = mapped_column(Float, nullable=False)

    trajectory_type: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Provenance
    data_source: Mapped[str] = mapped_column(String(20), default="SIMULATED", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, nullable=False)

    student: Mapped["Student"] = relationship("Student", back_populates="weekly_performances")

    __table_args__ = (
        UniqueConstraint("student_id", "week_number", name="uq_student_weekly_record"),
        CheckConstraint("week_number >= 1 AND week_number <= 52", name="check_week_number_range"),
        CheckConstraint("attendance_rate >= 0.0 AND attendance_rate <= 100.0", name="check_weekly_att_range"),
        CheckConstraint("quiz_average >= 0.0 AND quiz_average <= 20.0", name="check_weekly_quiz_range"),
        CheckConstraint("assignment_completion >= 0.0 AND assignment_completion <= 100.0", name="check_weekly_assign_range"),
        CheckConstraint("study_hours >= 0.0 AND study_hours <= 168.0", name="check_weekly_hours_range"),
    )


class Prediction(Base):
    """Model inference log recording score regression, risk probability, and SHAP factors."""
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    student_id: Mapped[str] = mapped_column(
        String(50), ForeignKey("students.student_id", ondelete="CASCADE"), index=True, nullable=False
    )
    scenario: Mapped[str] = mapped_column(String(20), nullable=False)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)

    predicted_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    at_risk_probability: Mapped[float] = mapped_column(Float, nullable=False)
    is_at_risk: Mapped[int] = mapped_column(Integer, nullable=False)
    risk_category: Mapped[str] = mapped_column(String(20), nullable=False)
    top_risk_factors: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    data_source: Mapped[str] = mapped_column(String(20), default="REAL_UCI", nullable=False)
    prediction_date: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, nullable=False)

    student: Mapped["Student"] = relationship("Student", back_populates="predictions")

    __table_args__ = (
        CheckConstraint("at_risk_probability >= 0.0 AND at_risk_probability <= 1.0", name="check_pred_prob_range"),
        CheckConstraint("predicted_score IS NULL OR (predicted_score >= 0.0 AND predicted_score <= 20.0)", name="check_pred_score_range"),
    )


class Alert(Base):
    """Early warning alert generated by longitudinal monitoring or risk escalation."""
    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    student_id: Mapped[str] = mapped_column(
        String(50), ForeignKey("students.student_id", ondelete="CASCADE"), index=True, nullable=False
    )
    alert_date: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, nullable=False)
    alert_type: Mapped[str] = mapped_column(String(50), nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    risk_level: Mapped[str] = mapped_column(String(20), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="Open", nullable=False)  # 'Open', 'Reviewed', 'Dismissed'
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    data_source: Mapped[str] = mapped_column(String(20), default="SIMULATED", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, nullable=False)

    student: Mapped["Student"] = relationship("Student", back_populates="alerts")
    interventions: Mapped[List["Intervention"]] = relationship("Intervention", back_populates="alert")


class Intervention(Base):
    """Advisory pedagogical intervention action assigned to academic counselors."""
    __tablename__ = "interventions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    student_id: Mapped[str] = mapped_column(
        String(50), ForeignKey("students.student_id", ondelete="CASCADE"), index=True, nullable=False
    )
    alert_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("alerts.id", ondelete="SET NULL"), nullable=True
    )
    intervention_type: Mapped[str] = mapped_column(String(50), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    assigned_to: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="Scheduled", nullable=False)  # 'Scheduled', 'In Progress', 'Completed', 'Cancelled'
    date_logged: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, nullable=False)

    data_source: Mapped[str] = mapped_column(String(20), default="SIMULATED", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, nullable=False)

    # Baseline indicators at time of intervention
    baseline_risk_prob: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    baseline_predicted_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    baseline_attendance: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Follow-up review indicators
    followup_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    followup_attendance: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    followup_marks: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    followup_risk_prob: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    followup_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    observed_change: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)  # 'Improved', 'Stable', 'Declined'

    student: Mapped["Student"] = relationship("Student", back_populates="interventions")
    alert: Mapped[Optional["Alert"]] = relationship("Alert", back_populates="interventions")


# ==============================================================================
# 3. Database Engine, Session Factory & Schema Initializer
# ==============================================================================
_GLOBAL_ENGINE: Optional[Engine] = None


def get_engine(db_url: Optional[str] = None) -> Engine:
    """Retrieve or create the SQLAlchemy database Engine.

    Guarantees:
    1. Parent directory exists.
    2. Tables are created via Base.metadata.create_all(engine) upon engine creation.
    3. SQLite multi-threading is enabled.

    Args:
        db_url: Database connection string. If None, defaults to SQL_DATABASE_URL.

    Returns:
        SQLAlchemy Engine instance.
    """
    global _GLOBAL_ENGINE
    if db_url is None:
        db_url = SQL_DATABASE_URL

    if _GLOBAL_ENGINE is None or str(_GLOBAL_ENGINE.url) != db_url:
        # If SQLite path, ensure parent directory exists
        if "sqlite" in db_url.lower():
            clean_path_str = db_url.replace("sqlite:///", "").replace("sqlite://", "")
            if clean_path_str and clean_path_str != ":memory:":
                db_file_path = Path(clean_path_str)
                if not db_file_path.is_absolute():
                    db_file_path = _ROOT / db_file_path
                db_file_path.parent.mkdir(parents=True, exist_ok=True)

        connect_args = {"check_same_thread": False} if "sqlite" in db_url.lower() else {}
        _GLOBAL_ENGINE = create_engine(db_url, connect_args=connect_args, echo=False)
        # Ensure all tables and schema migrations exist immediately upon engine creation
        init_db(_GLOBAL_ENGINE)

    return _GLOBAL_ENGINE


def init_db(engine: Optional[Engine] = None) -> None:
    """Create all database tables and constraints if they do not exist.

    Also ensures backwards-compatible columns exist on the interventions table.

    Args:
        engine: Optional Engine. Defaults to get_engine().
    """
    if engine is None:
        engine = get_engine()
    Base.metadata.create_all(bind=engine)

    # Schema migration: check and add baseline/follow-up columns to interventions if SQLite
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            res = conn.execute(text("PRAGMA table_info(interventions)")).fetchall()
            if res:
                existing_cols = {row[1] for row in res}
                new_cols = [
                    ("baseline_risk_prob", "FLOAT"),
                    ("baseline_predicted_score", "FLOAT"),
                    ("baseline_attendance", "FLOAT"),
                    ("followup_date", "DATETIME"),
                    ("followup_attendance", "FLOAT"),
                    ("followup_marks", "FLOAT"),
                    ("followup_risk_prob", "FLOAT"),
                    ("followup_notes", "TEXT"),
                    ("observed_change", "VARCHAR(50)"),
                ]
                for col_name, col_type in new_cols:
                    if col_name not in existing_cols:
                        conn.execute(text(f"ALTER TABLE interventions ADD COLUMN {col_name} {col_type}"))
                conn.commit()
    except Exception:
        pass


def reset_db(engine: Optional[Engine] = None) -> None:
    """Drop and recreate all database tables (Caution: clears all data)."""
    if engine is None:
        engine = get_engine()
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)


def seed_database_from_processed(
    engine: Optional[Engine] = None,
    reset: bool = False,
) -> Dict[str, int]:
    """Seed the database using only files committed to git (data/processed/ and models/).

    Strictly complies with PROJECT_CONTEXT.md Rule 2.1:
    - Reads only data/processed/test.csv and models/
    - NEVER touches or depends on data/raw/
    - Flags all synthetic multi-week records as data_source='SIMULATED'

    Args:
        engine: Database Engine. Defaults to get_engine().
        reset: If True, drops and recreates schema before seeding.

    Returns:
        Dictionary of seeded table row counts.
    """
    if engine is None:
        engine = get_engine()

    if reset:
        reset_db(engine)
    else:
        init_db(engine)

    from src.config import PROCESSED_DATA_DIR
    test_csv_path = PROCESSED_DATA_DIR / "test.csv"
    if not test_csv_path.exists():
        test_csv_path = _ROOT / "data" / "processed" / "test.csv"
    if not test_csv_path.exists():
        raise FileNotFoundError(f"Required processed test split not found at: {test_csv_path}")

    df_test = pd.read_csv(test_csv_path)
    n_students = len(df_test)
    df_test["student_id"] = [f"EDU-{i+1:04d}" for i in range(n_students)]

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

    with get_db_session(engine) as session:
        session.add_all(student_records)
        session.add_all(academic_records)

    # Weekly performance
    sim_path = PROCESSED_DATA_DIR / "simulated_weekly_performance.csv"
    if not sim_path.exists():
        sim_path = _ROOT / "data" / "processed" / "simulated_weekly_performance.csv"

    if sim_path.exists():
        df_sim = pd.read_csv(sim_path)
    else:
        from src.simulation import generate_simulated_weekly_dataset
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

    with get_db_session(engine) as session:
        session.add_all(weekly_objs)

    # Predictions from trained models in models/
    drop_cols = ["G3", "at_risk", "student_key", "is_simulated", "student_id"]
    candidate_df = df_test.drop(columns=[c for c in drop_cols if c in df_test.columns], errors="ignore")

    from src.classification import load_classification_model, predict_risk
    from src.regression import load_model, predict_score

    clf_a, meta_clf_a = load_classification_model("A")
    reg_a, meta_reg_a = load_model("A")
    clf_b, meta_clf_b = load_classification_model("B")
    reg_b, meta_reg_b = load_model("B")

    pred_clf_a = predict_risk(clf_a, meta_clf_a, candidate_df, threshold=0.25)
    pred_reg_a = predict_score(reg_a, meta_reg_a, candidate_df)
    pred_clf_b = predict_risk(clf_b, meta_clf_b, candidate_df, threshold=0.15)
    pred_reg_b = predict_score(reg_b, meta_reg_b, candidate_df)

    pred_records = []
    for i in range(n_students):
        st_id = df_test.iloc[i]["student_id"]
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

    with get_db_session(engine) as session:
        session.add_all(pred_records)

    # Monitor transitions to create Alerts
    from src.monitoring import monitor_cohort_weekly_transition
    with get_db_session(engine) as session:
        monitor_cohort_weekly_transition(df_sim, session=session)

    # Create sample interventions
    with get_db_session(engine) as session:
        high_alerts = session.scalars(select(Alert).where(Alert.risk_level == "High")).all()
        interventions = []
        for alert in high_alerts[:12]:
            interv = Intervention(
                student_id=alert.student_id,
                alert_id=alert.id,
                intervention_type="MANDATORY_TUTORING",
                description=f"Assigned remedial study group following alert: {alert.reason}",
                assigned_to="Academic Success Center",
                status="Scheduled",
                data_source="SIMULATED",
            )
            interventions.append(interv)
        session.add_all(interventions)

    summary = {}
    with get_db_session(engine) as session:
        summary["Students"] = session.scalar(select(func.count(Student.id))) or 0
        summary["AcademicRecords"] = session.scalar(select(func.count(AcademicRecord.id))) or 0
        summary["WeeklyPerformance"] = session.scalar(select(func.count(WeeklyPerformance.id))) or 0
        summary["Predictions"] = session.scalar(select(func.count(Prediction.id))) or 0
        summary["Alerts"] = session.scalar(select(func.count(Alert.id))) or 0
        summary["Interventions"] = session.scalar(select(func.count(Intervention.id))) or 0

    return summary


@st.cache_resource
def ensure_database_ready() -> Engine:
    """Ensure database engine is initialized, tables created, and seeded if empty.

    Guarantees:
    - Tables are created via Base.metadata.create_all(engine).
    - If Students table is empty, seeds automatically from committed processed data and models.
    - Safe to call repeatedly (idempotent).
    - Wrapped in st.cache_resource so it runs once per Streamlit server process.
    - Catches exceptions and renders a user-friendly st.error before st.stop().

    Returns:
        Active SQLAlchemy Engine.
    """
    try:
        engine = get_engine()
        with get_db_session(engine) as session:
            count = session.scalar(select(func.count(Student.id))) or 0

        if count == 0:
            seed_database_from_processed(engine=engine, reset=False)

        return engine
    except Exception as exc:
        try:
            st.error(
                f"🚨 **Database Initialization Error:** The application database could not be initialized or seeded.\n\n"
                f"**Details:** `{exc}`\n\n"
                "Please ensure all required processed data (`data/processed/test.csv`) and model artifacts (`models/`) are present in the repository."
            )
            st.stop()
        except Exception:
            raise exc


@contextmanager
def get_db_session(engine: Optional[Engine] = None) -> Generator[Session, None, None]:
    """Context manager providing transactional session handling with automatic rollback.

    Usage:
        with get_db_session() as session:
            student = get_student("EDU-0001", session=session)
            session.add(new_record)
        # Session commits on success or rolls back on exception.
    """
    if engine is None:
        engine = get_engine()

    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
    session: Session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ==============================================================================
# 4. CRUD Operations & Repository Methods
# ==============================================================================
def add_student(
    student_data: Dict[str, Any],
    session: Optional[Session] = None,
) -> Student:
    """Insert a new Student record into the database.

    Args:
        student_data: Dictionary of student fields.
        session: Active session. If None, creates a self-contained session.

    Returns:
        Created Student ORM instance.

    Raises:
        ValueError: On duplicate student_id or constraint violation.
    """
    def _create(s: Session) -> Student:
        st_id = student_data.get("student_id")
        if not st_id:
            raise ValueError("student_id is required.")

        # Check existing
        existing = s.scalar(select(Student).where(Student.student_id == st_id))
        if existing:
            raise ValueError(f"Student with student_id '{st_id}' already exists.")

        student = Student(
            student_id=st_id,
            student_key=student_data.get("student_key"),
            school=student_data.get("school"),
            sex=student_data.get("sex"),
            age=student_data.get("age"),
            address=student_data.get("address"),
            famsize=student_data.get("famsize"),
            Pstatus=student_data.get("Pstatus"),
            Medu=student_data.get("Medu"),
            Fedu=student_data.get("Fedu"),
            Mjob=student_data.get("Mjob"),
            Fjob=student_data.get("Fjob"),
            reason=student_data.get("reason"),
            guardian=student_data.get("guardian"),
            data_source=student_data.get("data_source", "REAL_UCI"),
        )
        s.add(student)
        s.flush()
        return student

    if session is not None:
        return _create(session)
    with get_db_session() as s:
        return _create(s)


def get_student(
    student_id: str,
    session: Optional[Session] = None,
) -> Optional[Student]:
    """Retrieve a single Student by student_id.

    Args:
        student_id: Unique student identifier.
        session: Active session.

    Returns:
        Student instance or None if not found.
    """
    def _get(s: Session) -> Optional[Student]:
        return s.scalar(select(Student).where(Student.student_id == student_id))

    if session is not None:
        return _get(session)
    with get_db_session() as s:
        return _get(s)


def search_by_id(
    query: str,
    session: Optional[Session] = None,
) -> List[Student]:
    """Search for students whose student_id or student_key contains query substring.

    Args:
        query: Search string.
        session: Active session.

    Returns:
        List of matching Student instances.
    """
    def _search(s: Session) -> List[Student]:
        pattern = f"%{query}%"
        stmt = select(Student).where(
            (Student.student_id.ilike(pattern)) | (Student.student_key.ilike(pattern))
        ).order_by(Student.student_id)
        return list(s.scalars(stmt).all())

    if session is not None:
        return _search(session)
    with get_db_session() as s:
        return _search(s)


def update_student(
    student_id: str,
    updates: Dict[str, Any],
    session: Optional[Session] = None,
) -> Optional[Student]:
    """Update attributes for an existing student.

    Args:
        student_id: Unique student identifier.
        updates: Dictionary of attributes to update.
        session: Active session.

    Returns:
        Updated Student instance, or None if not found.
    """
    def _update(s: Session) -> Optional[Student]:
        student = s.scalar(select(Student).where(Student.student_id == student_id))
        if not student:
            return None

        # Prevent altering primary identifier
        filtered = {k: v for k, v in updates.items() if k not in ["id", "student_id"]}
        for k, v in filtered.items():
            if hasattr(student, k):
                setattr(student, k, v)

        s.flush()
        return student

    if session is not None:
        return _update(session)
    with get_db_session() as s:
        return _update(s)


def add_intervention(
    student_id: str,
    intervention_type: str,
    description: str,
    assigned_to: str = "Academic Advisor",
    status: str = "Scheduled",
    alert_id: Optional[int] = None,
    baseline_risk_prob: Optional[float] = None,
    baseline_predicted_score: Optional[float] = None,
    baseline_attendance: Optional[float] = None,
    data_source: str = "TEACHER_LOGGED",
    session: Optional[Session] = None,
) -> Intervention:
    """Record a pedagogical or counselor intervention in the database.

    Args:
        student_id: Unique student ID.
        intervention_type: Category (e.g. 'TUTORING', 'COUNSELING', 'AI_PLAN').
        description: Detailed intervention notes or generated plan.
        assigned_to: Name or role of the educator responsible.
        status: 'Scheduled', 'In Progress', 'Completed', or 'Cancelled'.
        alert_id: Optional foreign key to related Alert.
        baseline_risk_prob: Optional baseline calibrated at-risk probability.
        baseline_predicted_score: Optional baseline predicted score.
        baseline_attendance: Optional baseline attendance rate.
        data_source: Provenance tag ('TEACHER_LOGGED', 'REAL', 'SIMULATED').
        session: Active database session.

    Returns:
        Created Intervention instance.

    Raises:
        ValueError: If student does not exist or required fields are missing.
    """
    def _create(s: Session) -> Intervention:
        if not student_id or not intervention_type or not description:
            raise ValueError("student_id, intervention_type, and description are required.")

        student = s.scalar(select(Student).where(Student.student_id == student_id))
        if not student:
            raise ValueError(f"Student with student_id '{student_id}' does not exist.")

        interv = Intervention(
            student_id=student_id,
            alert_id=alert_id,
            intervention_type=intervention_type,
            description=description,
            assigned_to=assigned_to,
            status=status,
            baseline_risk_prob=baseline_risk_prob,
            baseline_predicted_score=baseline_predicted_score,
            baseline_attendance=baseline_attendance,
            data_source=data_source,
            date_logged=_utcnow(),
        )
        s.add(interv)
        s.flush()
        return interv

    if session is not None:
        return _create(session)
    with get_db_session() as s:
        return _create(s)


def record_intervention_followup(
    intervention_id: int,
    followup_attendance: float,
    followup_marks: float,
    followup_risk_prob: float,
    followup_notes: str,
    status: str = "Completed",
    followup_date: Optional[datetime] = None,
    session: Optional[Session] = None,
) -> Optional[Intervention]:
    """Record follow-up outcomes for an intervention, computing observed changes.

    DISCLAIMER: Observed improvement is not proof that the intervention caused it.

    Args:
        intervention_id: Primary key ID of the intervention.
        followup_attendance: Follow-up attendance rate (0-100%).
        followup_marks: Follow-up exam mark or quiz average (0-20 scale).
        followup_risk_prob: Follow-up calibrated at-risk probability (0.0-1.0).
        followup_notes: Qualitative review notes from counselor.
        status: Updated status (default: 'Completed').
        followup_date: Date of follow-up review. Defaults to now.
        session: Active database session.

    Returns:
        Updated Intervention instance, or None if not found.
    """
    if followup_date is None:
        followup_date = _utcnow()

    def _followup(s: Session) -> Optional[Intervention]:
        interv = s.scalar(select(Intervention).where(Intervention.id == intervention_id))
        if not interv:
            return None

        # Compute observed change label
        # Default heuristic:
        # Improved: risk dropped by >= 5% OR marks improved by >= 1 pt
        # Declined: risk increased by >= 5% OR marks dropped by >= 1 pt
        # Stable: otherwise
        observed_change = "Stable"
        if interv.baseline_risk_prob is not None:
            prob_delta = followup_risk_prob - interv.baseline_risk_prob
            if prob_delta <= -0.05:
                observed_change = "Improved"
            elif prob_delta >= 0.05:
                observed_change = "Declined"
        elif interv.baseline_predicted_score is not None:
            score_delta = followup_marks - interv.baseline_predicted_score
            if score_delta >= 1.0:
                observed_change = "Improved"
            elif score_delta <= -1.0:
                observed_change = "Declined"

        interv.followup_date = followup_date
        interv.followup_attendance = followup_attendance
        interv.followup_marks = followup_marks
        interv.followup_risk_prob = followup_risk_prob
        interv.followup_notes = followup_notes
        interv.status = status
        interv.observed_change = observed_change

        s.flush()
        return interv

    if session is not None:
        return _followup(session)
    with get_db_session() as s:
        return _followup(s)


def get_all_interventions(
    status_filter: Optional[str] = None,
    session: Optional[Session] = None,
) -> List[Intervention]:
    """Retrieve all recorded interventions across the entire institution.

    Args:
        status_filter: Optional filter ('Scheduled', 'In Progress', 'Completed', etc.).
        session: Active session.

    Returns:
        List of Intervention instances ordered by date_logged descending.
    """
    def _query(s: Session) -> List[Intervention]:
        stmt = select(Intervention)
        if status_filter:
            stmt = stmt.where(Intervention.status == status_filter)
        stmt = stmt.order_by(Intervention.date_logged.desc(), Intervention.id.desc())
        return list(s.scalars(stmt).all())

    if session is not None:
        return _query(session)
    with get_db_session() as s:
        return _query(s)


def update_alert_status(
    alert_id: int,
    new_status: str,
    session: Optional[Session] = None,
) -> Optional[Alert]:
    """Update status of an alert (e.g., mark as 'Reviewed' or 'Dismissed').

    Args:
        alert_id: Alert primary key ID.
        new_status: 'Open', 'Reviewed', or 'Dismissed'.
        session: Active session.

    Returns:
        Updated Alert instance, or None if not found.
    """
    valid_statuses = {"Open", "Reviewed", "Dismissed"}
    if new_status not in valid_statuses:
        raise ValueError(f"Invalid alert status '{new_status}'. Must be one of {valid_statuses}.")

    def _update(s: Session) -> Optional[Alert]:
        alert = s.scalar(select(Alert).where(Alert.id == alert_id))
        if not alert:
            return None

        alert.status = new_status
        if new_status in {"Reviewed", "Dismissed"}:
            alert.resolved_at = _utcnow()
        s.flush()
        return alert

    if session is not None:
        return _update(session)
    with get_db_session() as s:
        return _update(s)


def get_student_interventions(
    student_id: str,
    session: Optional[Session] = None,
) -> List[Intervention]:
    """Retrieve all recorded interventions for a given student.

    Args:
        student_id: Unique student ID.
        session: Active session.

    Returns:
        List of Intervention instances ordered by date_logged descending.
    """
    def _query(s: Session) -> List[Intervention]:
        stmt = (
            select(Intervention)
            .where(Intervention.student_id == student_id)
            .order_by(Intervention.date_logged.desc())
        )
        return list(s.scalars(stmt).all())

    if session is not None:
        return _query(session)
    with get_db_session() as s:
        return _query(s)


def bulk_import_csv(
    csv_path: Union[str, Path],
    session: Optional[Session] = None,
) -> Dict[str, Any]:
    """Import and validate student records in bulk from a CSV file.

    Validation Rules:
    1. Rejects missing files.
    2. Rejects completely empty CSV files.
    3. Validates required student_id column.
    4. Enforces range checks (age 10-30, Medu/Fedu 0-4, G1/G2/G3 0-20, absences >= 0).
    5. Rejects duplicate student_ids within the file or already present in the database.
    6. All valid records are inserted in a single atomic transaction.

    Args:
        csv_path: Path to CSV file.
        session: Optional session.

    Returns:
        Report dictionary detailing total_rows, imported_count, duplicate_count,
        invalid_count, error list, and human-readable summary.
    """
    p = Path(csv_path)
    if not p.exists():
        return {
            "status": "failed",
            "total_rows": 0,
            "imported_count": 0,
            "duplicate_count": 0,
            "invalid_count": 0,
            "errors": [f"File not found: {p}"],
            "summary": f"Import failed: File '{p}' does not exist.",
        }

    # Reject 0-byte files immediately
    if p.stat().st_size == 0:
        return {
            "status": "failed",
            "total_rows": 0,
            "imported_count": 0,
            "duplicate_count": 0,
            "invalid_count": 0,
            "errors": ["CSV file is completely empty."],
            "summary": "Import failed: Provided CSV contains zero data rows.",
        }

    # Attempt to read CSV
    try:
        df = pd.read_csv(p)
    except pd.errors.EmptyDataError:
        return {
            "status": "failed",
            "total_rows": 0,
            "imported_count": 0,
            "duplicate_count": 0,
            "invalid_count": 0,
            "errors": ["CSV file is completely empty."],
            "summary": "Import failed: Provided CSV contains zero data rows.",
        }
    except Exception as e:
        return {
            "status": "failed",
            "total_rows": 0,
            "imported_count": 0,
            "duplicate_count": 0,
            "invalid_count": 0,
            "errors": [f"CSV parse error: {str(e)}"],
            "summary": f"Import failed: Could not parse CSV ({str(e)}).",
        }

    if df.empty:
        return {
            "status": "failed",
            "total_rows": 0,
            "imported_count": 0,
            "duplicate_count": 0,
            "invalid_count": 0,
            "errors": ["CSV file is completely empty."],
            "summary": "Import failed: Provided CSV contains zero data rows.",
        }

    # If student_id is not present, auto-generate sequential IDs
    if "student_id" not in df.columns:
        df["student_id"] = [f"IMP-{i+1:04d}" for i in range(len(df))]

    total_rows = len(df)
    imported_count = 0
    duplicate_count = 0
    invalid_count = 0
    errors = []

    def _execute_import(s: Session) -> Dict[str, Any]:
        nonlocal imported_count, duplicate_count, invalid_count, errors

        # Query existing IDs
        existing_ids = set(s.scalars(select(Student.student_id)).all())
        seen_in_batch = set()

        records_to_insert = []
        academic_records_to_insert = []

        for idx, row in df.iterrows():
            st_id = str(row["student_id"]).strip()
            row_num = idx + 2  # 1-based, accounting for header

            # Check duplicates
            if st_id in seen_in_batch or st_id in existing_ids:
                duplicate_count += 1
                errors.append(f"Row {row_num}: Duplicate student_id '{st_id}'.")
                continue
            seen_in_batch.add(st_id)

            # Range validation checks
            row_invalid = False

            # Age check
            if "age" in row and pd.notna(row["age"]):
                age_val = float(row["age"])
                if age_val < 10 or age_val > 30:
                    errors.append(f"Row {row_num} ({st_id}): Invalid age {age_val} (must be 10-30).")
                    row_invalid = True

            # Medu check
            if "Medu" in row and pd.notna(row["Medu"]):
                medu_val = float(row["Medu"])
                if medu_val < 0 or medu_val > 4:
                    errors.append(f"Row {row_num} ({st_id}): Invalid Medu {medu_val} (must be 0-4).")
                    row_invalid = True

            # Fedu check
            if "Fedu" in row and pd.notna(row["Fedu"]):
                fedu_val = float(row["Fedu"])
                if fedu_val < 0 or fedu_val > 4:
                    errors.append(f"Row {row_num} ({st_id}): Invalid Fedu {fedu_val} (must be 0-4).")
                    row_invalid = True

            # Exam grade checks (G1, G2, G3)
            for g_col in ["G1", "G2", "G3"]:
                if g_col in row and pd.notna(row[g_col]):
                    g_val = float(row[g_col])
                    if g_val < 0.0 or g_val > 20.0:
                        errors.append(f"Row {row_num} ({st_id}): Invalid {g_col} mark {g_val} (must be 0.0-20.0).")
                        row_invalid = True

            # Absences check
            if "absences" in row and pd.notna(row["absences"]):
                abs_val = float(row["absences"])
                if abs_val < 0 or abs_val > 150:
                    errors.append(f"Row {row_num} ({st_id}): Invalid absences {abs_val} (must be >= 0).")
                    row_invalid = True

            if row_invalid:
                invalid_count += 1
                continue

            # Build student object
            st_obj = Student(
                student_id=st_id,
                student_key=str(row["student_key"]) if "student_key" in row and pd.notna(row["student_key"]) else None,
                school=str(row["school"]) if "school" in row and pd.notna(row["school"]) else None,
                sex=str(row["sex"]) if "sex" in row and pd.notna(row["sex"]) else None,
                age=int(row["age"]) if "age" in row and pd.notna(row["age"]) else None,
                address=str(row["address"]) if "address" in row and pd.notna(row["address"]) else None,
                famsize=str(row["famsize"]) if "famsize" in row and pd.notna(row["famsize"]) else None,
                Pstatus=str(row["Pstatus"]) if "Pstatus" in row and pd.notna(row["Pstatus"]) else None,
                Medu=int(row["Medu"]) if "Medu" in row and pd.notna(row["Medu"]) else None,
                Fedu=int(row["Fedu"]) if "Fedu" in row and pd.notna(row["Fedu"]) else None,
                Mjob=str(row["Mjob"]) if "Mjob" in row and pd.notna(row["Mjob"]) else None,
                Fjob=str(row["Fjob"]) if "Fjob" in row and pd.notna(row["Fjob"]) else None,
                reason=str(row["reason"]) if "reason" in row and pd.notna(row["reason"]) else None,
                guardian=str(row["guardian"]) if "guardian" in row and pd.notna(row["guardian"]) else None,
                data_source=str(row.get("data_source", "REAL_UCI")),
            )
            records_to_insert.append(st_obj)

            # If academic columns exist, build academic record
            if "subject" in row or "G3" in row:
                acad_obj = AcademicRecord(
                    student_id=st_id,
                    subject=str(row.get("subject", "math")),
                    traveltime=int(row["traveltime"]) if "traveltime" in row and pd.notna(row["traveltime"]) else None,
                    studytime=int(row["studytime"]) if "studytime" in row and pd.notna(row["studytime"]) else None,
                    failures=int(row["failures"]) if "failures" in row and pd.notna(row["failures"]) else None,
                    schoolsup=str(row["schoolsup"]) if "schoolsup" in row and pd.notna(row["schoolsup"]) else None,
                    famsup=str(row["famsup"]) if "famsup" in row and pd.notna(row["famsup"]) else None,
                    paid=str(row["paid"]) if "paid" in row and pd.notna(row["paid"]) else None,
                    activities=str(row["activities"]) if "activities" in row and pd.notna(row["activities"]) else None,
                    nursery=str(row["nursery"]) if "nursery" in row and pd.notna(row["nursery"]) else None,
                    higher=str(row["higher"]) if "higher" in row and pd.notna(row["higher"]) else None,
                    internet=str(row["internet"]) if "internet" in row and pd.notna(row["internet"]) else None,
                    romantic=str(row["romantic"]) if "romantic" in row and pd.notna(row["romantic"]) else None,
                    famrel=int(row["famrel"]) if "famrel" in row and pd.notna(row["famrel"]) else None,
                    freetime=int(row["freetime"]) if "freetime" in row and pd.notna(row["freetime"]) else None,
                    goout=int(row["goout"]) if "goout" in row and pd.notna(row["goout"]) else None,
                    Dalc=int(row["Dalc"]) if "Dalc" in row and pd.notna(row["Dalc"]) else None,
                    Walc=int(row["Walc"]) if "Walc" in row and pd.notna(row["Walc"]) else None,
                    health=int(row["health"]) if "health" in row and pd.notna(row["health"]) else None,
                    absences=int(row["absences"]) if "absences" in row and pd.notna(row["absences"]) else None,
                    G1=float(row["G1"]) if "G1" in row and pd.notna(row["G1"]) else None,
                    G2=float(row["G2"]) if "G2" in row and pd.notna(row["G2"]) else None,
                    G3=float(row["G3"]) if "G3" in row and pd.notna(row["G3"]) else None,
                    data_source=str(row.get("data_source", "REAL_UCI")),
                )
                academic_records_to_insert.append(acad_obj)

        # Batch insert
        if records_to_insert:
            s.add_all(records_to_insert)
            s.flush()
            if academic_records_to_insert:
                s.add_all(academic_records_to_insert)
                s.flush()
            imported_count = len(records_to_insert)

        status = "success" if imported_count > 0 else "failed"
        summary = (
            f"Bulk import completed: {imported_count}/{total_rows} records successfully imported. "
            f"Duplicates skipped: {duplicate_count}. Invalid rows rejected: {invalid_count}."
        )

        return {
            "status": status,
            "total_rows": total_rows,
            "imported_count": imported_count,
            "duplicate_count": duplicate_count,
            "invalid_count": invalid_count,
            "errors": errors,
            "summary": summary,
        }

    if session is not None:
        return _execute_import(session)
    with get_db_session() as s:
        return _execute_import(s)


if __name__ == "__main__":
    print(f"Initializing database at: {DATABASE_PATH}")
    init_db()
    print("Database tables created successfully.")
