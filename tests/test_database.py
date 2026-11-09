"""
Unit Tests for EduGuard AI Database, Persistence, and CRUD Layer (Stage 6)
"""

import tempfile
from pathlib import Path
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from src.database import (
    AcademicRecord,
    Alert,
    Base,
    Prediction,
    Student,
    WeeklyPerformance,
    add_student,
    bulk_import_csv,
    get_db_session,
    get_student,
    init_db,
    search_by_id,
    update_student,
)


@pytest.fixture
def mem_engine():
    """Create an isolated in-memory SQLite engine with PRAGMA foreign keys enabled."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    # Enable foreign keys
    with engine.connect() as conn:
        conn.exec_driver_sql("PRAGMA foreign_keys=ON")
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def mem_session(mem_engine):
    """Provide a transactional session bound to the in-memory engine."""
    SessionLocal = sessionmaker(bind=mem_engine, expire_on_commit=False)
    session = SessionLocal()
    yield session
    session.close()


def test_insert_and_retrieve_student(mem_session):
    """Verify adding a student persists and retrieves all demographic fields accurately."""
    student_data = {
        "student_id": "TEST-001",
        "student_key": "GP_M_16_U_LE3_T_4_4",
        "school": "GP",
        "sex": "M",
        "age": 16,
        "address": "U",
        "famsize": "LE3",
        "Pstatus": "T",
        "Medu": 4,
        "Fedu": 4,
        "Mjob": "services",
        "Fjob": "teacher",
        "reason": "course",
        "guardian": "mother",
        "data_source": "REAL_UCI",
    }
    created = add_student(student_data, session=mem_session)
    assert created.id is not None
    assert created.student_id == "TEST-001"

    # Retrieve
    retrieved = get_student("TEST-001", session=mem_session)
    assert retrieved is not None
    assert retrieved.student_id == "TEST-001"
    assert retrieved.age == 16
    assert retrieved.school == "GP"
    assert retrieved.data_source == "REAL_UCI"


def test_duplicate_student_id_raises_error(mem_session):
    """Verify unique constraint on student_id prevents duplicates."""
    student_data = {"student_id": "TEST-DUP", "age": 17}
    add_student(student_data, session=mem_session)

    # Attempt second insertion with same student_id
    with pytest.raises(ValueError, match="already exists"):
        add_student(student_data, session=mem_session)


def test_invalid_marks_constraint_raises_error(mem_session):
    """Verify CheckConstraint rejects marks outside the Portuguese 0-20 scale."""
    st = Student(student_id="TEST-MARKS", age=16, data_source="REAL_UCI")
    mem_session.add(st)
    mem_session.flush()

    # Invalid G3 = 25.0 (> 20.0)
    acad_invalid = AcademicRecord(
        student_id="TEST-MARKS",
        subject="math",
        G1=10.0,
        G2=12.0,
        G3=25.0,  # Violates check_academic_g3_range
        data_source="REAL_UCI",
    )
    mem_session.add(acad_invalid)

    with pytest.raises(IntegrityError):
        mem_session.flush()

    mem_session.rollback()

    # Invalid negative mark G1 = -3.0 (< 0.0)
    acad_negative = AcademicRecord(
        student_id="TEST-MARKS",
        subject="math",
        G1=-3.0,
        G2=10.0,
        G3=10.0,
        data_source="REAL_UCI",
    )
    mem_session.add(acad_negative)
    with pytest.raises(IntegrityError):
        mem_session.flush()


def test_invalid_age_constraint(mem_session):
    """Verify CheckConstraint rejects invalid student age (must be 10-30)."""
    st_too_young = Student(student_id="TEST-YOUNG", age=7, data_source="REAL_UCI")
    mem_session.add(st_too_young)
    with pytest.raises(IntegrityError):
        mem_session.flush()

    mem_session.rollback()

    st_too_old = Student(student_id="TEST-OLD", age=45, data_source="REAL_UCI")
    mem_session.add(st_too_old)
    with pytest.raises(IntegrityError):
        mem_session.flush()


def test_empty_csv_bulk_import(mem_session):
    """Verify bulk_import_csv rejects empty CSV files without raising unhandled exceptions."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        # Completely empty file
        pass

    try:
        res = bulk_import_csv(f.name, session=mem_session)
        assert res["status"] == "failed"
        assert res["total_rows"] == 0
        assert res["imported_count"] == 0
        assert "empty" in res["summary"].lower() or "empty" in str(res["errors"]).lower()
    finally:
        Path(f.name).unlink(missing_ok=True)


def test_bulk_import_validates_ranges_and_duplicates(mem_session):
    """Verify bulk_import_csv filters out duplicate IDs and out-of-range marks."""
    csv_content = """student_id,age,school,sex,subject,G1,G2,G3
VAL-001,16,GP,F,math,12.0,14.0,15.0
VAL-002,17,MS,M,portuguese,10.0,11.0,12.0
VAL-001,16,GP,F,math,12.0,14.0,15.0
BAD-AGE,5,GP,F,math,10.0,10.0,10.0
BAD-MARK,17,GP,M,math,10.0,10.0,28.0
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
        f.write(csv_content)

    try:
        report = bulk_import_csv(f.name, session=mem_session)
        assert report["status"] == "success"
        assert report["total_rows"] == 5
        assert report["imported_count"] == 2  # VAL-001 and VAL-002
        assert report["duplicate_count"] == 1  # Duplicate VAL-001
        assert report["invalid_count"] == 2    # BAD-AGE and BAD-MARK
        assert len(report["errors"]) == 3
    finally:
        Path(f.name).unlink(missing_ok=True)


def test_rollback_on_error(mem_engine):
    """Verify session helper completely rolls back transaction when an exception occurs."""
    SessionLocal = sessionmaker(bind=mem_engine, expire_on_commit=False)

    # Use context manager simulating transaction failure
    with pytest.raises(RuntimeError):
        with get_db_session(engine=mem_engine) as session:
            st = Student(student_id="ROLLBACK-TEST", age=16, data_source="REAL_UCI")
            session.add(st)
            session.flush()
            # Force unhandled exception
            raise RuntimeError("Simulated mid-transaction failure!")

    # Verify student was NOT persisted after exception
    with SessionLocal() as verify_session:
        retrieved = verify_session.scalar(select(Student).where(Student.student_id == "ROLLBACK-TEST"))
        assert retrieved is None, "Transaction should have rolled back completely!"


def test_search_by_id(mem_session):
    """Verify search_by_id finds students by partial substring match."""
    st1 = Student(student_id="CS-2024-001", student_key="GP_M_15", age=15, data_source="REAL_UCI")
    st2 = Student(student_id="CS-2024-002", student_key="MS_F_16", age=16, data_source="REAL_UCI")
    st3 = Student(student_id="ENG-2024-001", student_key="GP_F_17", age=17, data_source="REAL_UCI")
    mem_session.add_all([st1, st2, st3])
    mem_session.flush()

    results_cs = search_by_id("CS-2024", session=mem_session)
    assert len(results_cs) == 2
    assert {s.student_id for s in results_cs} == {"CS-2024-001", "CS-2024-002"}

    results_eng = search_by_id("ENG", session=mem_session)
    assert len(results_eng) == 1
    assert results_eng[0].student_id == "ENG-2024-001"


def test_update_student(mem_session):
    """Verify update_student updates mutable attributes and ignores primary keys."""
    st = Student(student_id="UPD-001", age=16, address="U", data_source="REAL_UCI")
    mem_session.add(st)
    mem_session.flush()

    updated = update_student("UPD-001", {"age": 17, "address": "R", "student_id": "TAMPERED"}, session=mem_session)
    assert updated is not None
    assert updated.age == 17
    assert updated.address == "R"
    assert updated.student_id == "UPD-001", "student_id must remain immutable."


def test_cascade_delete(mem_session):
    """Verify deleting a student cascades and deletes their child records."""
    st = Student(student_id="CASC-001", age=16, data_source="REAL_UCI")
    mem_session.add(st)
    mem_session.flush()

    acad = AcademicRecord(student_id="CASC-001", subject="math", G3=14.0, data_source="REAL_UCI")
    alert = Alert(student_id="CASC-001", alert_type="TEST", reason="Test alert", risk_level="Low", data_source="SIMULATED")
    mem_session.add_all([acad, alert])
    mem_session.flush()

    # Delete student
    mem_session.delete(st)
    mem_session.flush()

    # Child records must be deleted
    assert mem_session.scalar(select(AcademicRecord).where(AcademicRecord.student_id == "CASC-001")) is None
    assert mem_session.scalar(select(Alert).where(Alert.student_id == "CASC-001")) is None

