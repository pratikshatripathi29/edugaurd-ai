"""
Unit Tests for EduGuard AI Longitudinal Monitoring & Intervention Tracking (Stage 10).

Validates:
1. Early-warning alert triggers (risk escalation, probability jump, attendance drop, assignment drop).
2. Alert status resolution ("Mark Reviewed" updating resolved_at timestamp).
3. Intervention Tracker: baseline metric capture, follow-up evaluation, and observed change classification.
4. Adding a new week of data (Week 13+), automated ML re-scoring, and real-time alert generation.
5. Data provenance tags enforcing 'SIMULATED' labeling on synthetic records.
"""

from datetime import datetime, timezone
import pytest
from sqlalchemy import select

from src.database import (
    AcademicRecord,
    Alert,
    Intervention,
    Prediction,
    Student,
    WeeklyPerformance,
    add_intervention,
    get_db_session,
    record_intervention_followup,
)
from src.monitoring import (
    check_assignment_drop,
    check_attendance_drop,
    check_probability_jump,
    check_risk_category_escalation,
    create_alert,
    evaluate_student_weekly_monitoring,
    get_open_alerts,
    log_new_weekly_performance_and_rescore,
    update_alert_status,
)


# ==============================================================================
# 1. Alert Trigger Unit Tests
# ==============================================================================
def test_check_risk_category_escalation():
    """Verify risk escalation triggers when risk tier shifts upward."""
    # Low -> High (Escalation)
    alert = check_risk_category_escalation("Low", "High")
    assert alert is not None
    assert alert["alert_type"] == "RISK_ESCALATION"
    assert "escalated" in alert["reason"].lower()

    # Medium -> High (Escalation)
    alert_med = check_risk_category_escalation("Medium", "High")
    assert alert_med is not None

    # High -> Low (De-escalation: no alert)
    assert check_risk_category_escalation("High", "Low") is None

    # Same category (no alert)
    assert check_risk_category_escalation("Medium", "Medium") is None


def test_check_probability_jump():
    """Verify probability surge alert fires when delta exceeds threshold."""
    # Jump of +0.25 (exceeds default 0.15)
    alert = check_probability_jump(prev_prob=0.20, curr_prob=0.45, jump_threshold=0.15)
    assert alert is not None
    assert alert["alert_type"] == "PROBABILITY_JUMP"
    assert "+25.0%" in alert["reason"]

    # Jump of +0.05 (under threshold: no alert)
    assert check_probability_jump(prev_prob=0.20, curr_prob=0.25, jump_threshold=0.15) is None

    # Decline in probability (no alert)
    assert check_probability_jump(prev_prob=0.60, curr_prob=0.35, jump_threshold=0.15) is None


def test_check_attendance_drop():
    """Verify attendance floor breach and sharp drop triggers."""
    # Floor breach (< 75%)
    alert_floor = check_attendance_drop(curr_attendance=68.0, prev_attendance=72.0)
    assert alert_floor is not None
    assert alert_floor["alert_type"] == "ATTENDANCE_DROP"
    assert "below the institutional warning threshold" in alert_floor["reason"].lower()

    # Sharp drop from prior week (e.g. 95% -> 78%, drop >= 15%)
    alert_drop = check_attendance_drop(curr_attendance=78.0, prev_attendance=95.0, drop_threshold=15.0)
    assert alert_drop is not None
    assert "17.0%" in alert_drop["reason"]

    # Stable attendance (no alert)
    assert check_attendance_drop(curr_attendance=92.0, prev_attendance=94.0) is None


def test_check_assignment_drop():
    """Verify assignment completion floor and sharp drop triggers."""
    # Floor breach (< 60%)
    alert_floor = check_assignment_drop(curr_completion=50.0, prev_completion=58.0)
    assert alert_floor is not None
    assert alert_floor["alert_type"] == "ASSIGNMENT_DROP"

    # Sharp drop (e.g. 90% -> 65%, drop >= 20%)
    alert_drop = check_assignment_drop(curr_completion=65.0, prev_completion=90.0, drop_threshold=20.0)
    assert alert_drop is not None

    # Normal fluctuations (no alert)
    assert check_assignment_drop(curr_completion=85.0, prev_completion=88.0) is None


# ==============================================================================
# 2. Database Alert Resolution & Filtering Tests
# ==============================================================================
def test_create_and_update_alert_status():
    """Verify persisting alert, querying open alerts, and updating status to Reviewed."""
    with get_db_session() as s:
        target_student = s.scalar(select(Student.student_id))
        assert target_student is not None

    # Create test alert
    new_alert = create_alert(
        student_id=target_student,
        alert_type="TEST_DISTRESS",
        reason="Weekly quiz dropped significantly.",
        risk_level="High",
        status="Open",
        data_source="SIMULATED",
    )

    assert new_alert.id is not None
    assert new_alert.status == "Open"
    assert new_alert.data_source == "SIMULATED"

    # Resolve alert
    updated_alert = update_alert_status(new_alert.id, "Reviewed")
    assert updated_alert is not None
    assert updated_alert.status == "Reviewed"
    assert updated_alert.resolved_at is not None


def test_get_open_alerts_filtering():
    """Verify get_open_alerts retrieves records and applies filters."""
    open_alerts = get_open_alerts(min_risk_level="Medium")
    assert isinstance(open_alerts, list)
    for a in open_alerts:
        assert a.status == "Open"
        assert a.risk_level in {"Medium", "High"}


# ==============================================================================
# 3. Intervention Tracker & Outcomes Tests
# ==============================================================================
def test_intervention_tracker_baseline_and_followup_improved():
    """Verify logging an intervention with baseline metrics and recording improved follow-up."""
    with get_db_session() as s:
        student_id = s.scalar(select(Student.student_id))

    # 1. Log intervention with baseline indicators
    interv = add_intervention(
        student_id=student_id,
        intervention_type="MANDATORY_TUTORING",
        description="Assigned 2 hours per week peer study clinic.",
        assigned_to="Counselor Vance",
        status="Scheduled",
        baseline_risk_prob=0.72,
        baseline_predicted_score=7.5,
        baseline_attendance=65.0,
        data_source="TEACHER_LOGGED",
    )

    assert interv.id is not None
    assert interv.baseline_risk_prob == 0.72
    assert interv.baseline_predicted_score == 7.5

    # 2. Record follow-up with significant improvement (risk dropped from 0.72 to 0.35)
    updated = record_intervention_followup(
        intervention_id=interv.id,
        followup_attendance=90.0,
        followup_marks=12.5,
        followup_risk_prob=0.35,
        followup_notes="Student attended all tutoring sessions and showed mastery in quiz.",
        status="Completed",
    )

    assert updated is not None
    assert updated.status == "Completed"
    assert updated.followup_marks == 12.5
    assert updated.observed_change == "Improved"


def test_intervention_tracker_followup_declined():
    """Verify follow-up classifies as 'Declined' when risk increases."""
    with get_db_session() as s:
        student_id = s.scalar(select(Student.student_id))

    interv = add_intervention(
        student_id=student_id,
        intervention_type="ATTENDANCE_COUNSELING",
        description="Attendance contract discussion.",
        assigned_to="Counselor Vance",
        status="Scheduled",
        baseline_risk_prob=0.30,
        baseline_predicted_score=13.0,
        baseline_attendance=85.0,
        data_source="TEACHER_LOGGED",
    )

    # Follow-up showing higher risk (0.30 -> 0.65)
    updated = record_intervention_followup(
        intervention_id=interv.id,
        followup_attendance=60.0,
        followup_marks=9.0,
        followup_risk_prob=0.65,
        followup_notes="Missed further sessions due to personal scheduling conflict.",
        status="In Progress",
    )

    assert updated.observed_change == "Declined"


# ==============================================================================
# 4. Live Weekly Performance Logging & Automated Re-scoring Tests
# ==============================================================================
def test_log_new_weekly_performance_and_rescore():
    """Verify adding a new week of data (Week 13+), automated ML re-scoring, and alert generation."""
    with get_db_session() as s:
        student = s.scalar(select(Student))
        target_st_id = student.student_id

        # Query existing count of weeks
        pre_weeks = s.scalar(
            select(WeeklyPerformance.week_number)
            .where(WeeklyPerformance.student_id == target_st_id)
            .order_by(WeeklyPerformance.week_number.desc())
        ) or 0

    # Submit new week with severe distress to test re-scoring and alert trigger
    res = log_new_weekly_performance_and_rescore(
        student_id=target_st_id,
        attendance_rate=45.0,       # Severe drop < 75%
        quiz_average=4.0,           # Low quiz average
        assignment_completion=30.0, # Severe drop < 60%
        study_hours=1.0,            # Low study hours
        notes="Automated test week logging.",
        trajectory_type="DECLINING",
        data_source="SIMULATED",
    )

    assert res["status"] == "success"
    assert res["student_id"] == target_st_id
    assert res["week_number"] == pre_weeks + 1
    assert 0.0 <= res["new_predicted_score"] <= 20.0
    assert 0.0 <= res["new_risk_probability"] <= 1.0
    assert res["new_risk_tier"] in {"Low", "Medium", "High"}
    assert res["alerts_count"] >= 1  # Distress metrics must trigger alerts

    # Verify Prediction record in DB was updated
    with get_db_session() as s:
        pred_record = s.scalar(
            select(Prediction).where(
                Prediction.student_id == target_st_id,
                Prediction.scenario == "Scenario B (Mid-Semester, With G1/G2)",
            )
        )
        assert pred_record is not None
        assert pred_record.predicted_score == res["new_predicted_score"]
        assert pred_record.at_risk_probability == res["new_risk_probability"]
