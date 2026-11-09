"""
EduGuard AI — Longitudinal Early-Warning Monitoring Module (Stage 6)

This module tracks longitudinal academic indicators (attendance, quiz averages,
assignment completion, study hours, and model risk probability transitions) to
generate actionable early-warning alerts for school counselors.

Alert Generation Triggers:
1. Risk Category Escalation:
   - Low -> Medium
   - Low -> High
   - Medium -> High
2. Probability Surge:
   - Calibrated at-risk probability jumps by >= configurable threshold (default: +15%).
3. Attendance Distress:
   - Weekly attendance falls below critical floor (default: < 75%).
   - Sudden drop from prior week by >= threshold (default: >= 15%).
4. Assignment Disengagement:
   - Weekly assignment completion falls below critical floor (default: < 60%).
   - Sudden drop from prior week by >= threshold (default: >= 20%).

GOVERNANCE & PROVENANCE NOTICE (PROJECT_CONTEXT.md Rule 2.1):
Alerts generated from synthetic weekly logs are explicitly flagged with
`data_source = 'SIMULATED'`.
"""

from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path when invoked directly
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.config import RISK_THRESHOLDS
from src.database import (
    AcademicRecord,
    Alert,
    Prediction,
    Student,
    WeeklyPerformance,
    get_db_session,
)

# Risk tier order hierarchy
RISK_LEVEL_RANKS: Dict[str, int] = {
    "Low": 1,
    "Medium": 2,
    "High": 3,
}


def check_risk_category_escalation(
    prev_category: Optional[str],
    curr_category: str,
) -> Optional[Dict[str, Any]]:
    """Detect when a student's risk category escalates (e.g. Low -> Medium, Medium -> High).

    Args:
        prev_category: Previous week/evaluation risk tier ('Low', 'Medium', 'High').
        curr_category: Current week/evaluation risk tier.

    Returns:
        Alert dictionary if an escalation occurred; None otherwise.
    """
    if not prev_category or prev_category == curr_category:
        return None

    prev_rank = RISK_LEVEL_RANKS.get(prev_category, 1)
    curr_rank = RISK_LEVEL_RANKS.get(curr_category, 1)

    if curr_rank > prev_rank:
        return {
            "alert_type": "RISK_ESCALATION",
            "reason": (
                f"Academic risk tier escalated from '{prev_category}' to '{curr_category}'. "
                f"Urgent counselor triage advised."
            ),
            "risk_level": curr_category,
        }

    return None


def check_probability_jump(
    prev_prob: Optional[float],
    curr_prob: float,
    jump_threshold: float = 0.15,
) -> Optional[Dict[str, Any]]:
    """Detect when at-risk probability increases significantly between evaluations.

    Args:
        prev_prob: Previous probability [0.0, 1.0].
        curr_prob: Current probability [0.0, 1.0].
        jump_threshold: Minimum jump to trigger alert (default: 0.15 or +15%).

    Returns:
        Alert dictionary if jump exceeds threshold; None otherwise.
    """
    if prev_prob is None:
        return None

    diff = curr_prob - prev_prob
    if diff >= jump_threshold:
        risk_level = "High" if curr_prob >= RISK_THRESHOLDS["high"] else "Medium"
        return {
            "alert_type": "PROBABILITY_JUMP",
            "reason": (
                f"Calibrated at-risk probability surged by +{diff * 100:.1f}% "
                f"(from {prev_prob * 100:.1f}% to {curr_prob * 100:.1f}%)."
            ),
            "risk_level": risk_level,
        }

    return None


def check_attendance_drop(
    curr_attendance: float,
    prev_attendance: Optional[float] = None,
    floor_threshold: float = 75.0,
    drop_threshold: float = 15.0,
) -> Optional[Dict[str, Any]]:
    """Detect attendance distress: either falling below floor or dropping steeply.

    Args:
        curr_attendance: Current week attendance percentage (0.0 to 100.0).
        prev_attendance: Prior week attendance percentage.
        floor_threshold: Absolute attendance floor (default: 75.0%).
        drop_threshold: Single-week drop threshold (default: 15.0%).

    Returns:
        Alert dictionary if attendance criteria met; None otherwise.
    """
    # Check 1: Sudden single-week drop
    if prev_attendance is not None and (prev_attendance - curr_attendance) >= drop_threshold:
        return {
            "alert_type": "ATTENDANCE_DROP",
            "reason": (
                f"Attendance plummeted by {prev_attendance - curr_attendance:.1f}% in a single week "
                f"(from {prev_attendance:.1f}% to {curr_attendance:.1f}%)."
            ),
            "risk_level": "High" if curr_attendance < 60.0 else "Medium",
        }

    # Check 2: Absolute low attendance
    if curr_attendance < floor_threshold:
        return {
            "alert_type": "ATTENDANCE_DROP",
            "reason": (
                f"Weekly attendance fell to {curr_attendance:.1f}%, below the institutional "
                f"warning threshold of {floor_threshold:.1f}%."
            ),
            "risk_level": "High" if curr_attendance < 60.0 else "Medium",
        }

    return None


def check_assignment_drop(
    curr_completion: float,
    prev_completion: Optional[float] = None,
    floor_threshold: float = 60.0,
    drop_threshold: float = 20.0,
) -> Optional[Dict[str, Any]]:
    """Detect assignment completion collapse.

    Args:
        curr_completion: Current week assignment completion rate (0.0 to 100.0).
        prev_completion: Prior week assignment completion rate.
        floor_threshold: Absolute completion floor (default: 60.0%).
        drop_threshold: Single-week drop threshold (default: 20.0%).

    Returns:
        Alert dictionary if completion criteria met; None otherwise.
    """
    # Check 1: Sudden single-week drop
    if prev_completion is not None and (prev_completion - curr_completion) >= drop_threshold:
        return {
            "alert_type": "ASSIGNMENT_DROP",
            "reason": (
                f"Assignment completion dropped by {prev_completion - curr_completion:.1f}% "
                f"(from {prev_completion:.1f}% to {curr_completion:.1f}%)."
            ),
            "risk_level": "High" if curr_completion < 40.0 else "Medium",
        }

    # Check 2: Absolute low completion
    if curr_completion < floor_threshold:
        return {
            "alert_type": "ASSIGNMENT_DROP",
            "reason": (
                f"Weekly assignment completion is critically low at {curr_completion:.1f}% "
                f"(below {floor_threshold:.1f}% threshold)."
            ),
            "risk_level": "High" if curr_completion < 40.0 else "Medium",
        }

    return None


def evaluate_student_weekly_monitoring(
    student_id: str,
    curr_week_record: Dict[str, Any],
    prev_week_record: Optional[Dict[str, Any]] = None,
    prob_jump_threshold: float = 0.15,
    attendance_floor: float = 75.0,
    attendance_drop_threshold: float = 15.0,
    assignment_floor: float = 60.0,
    assignment_drop_threshold: float = 20.0,
) -> List[Dict[str, Any]]:
    """Evaluate all early-warning rules for a student on a specific academic week.

    Args:
        student_id: Unique student ID.
        curr_week_record: Performance dictionary for current week.
        prev_week_record: Performance dictionary for preceding week.
        prob_jump_threshold: Probability surge threshold.
        attendance_floor: Low attendance threshold.
        attendance_drop_threshold: Weekly attendance drop threshold.
        assignment_floor: Low assignment completion floor.
        assignment_drop_threshold: Weekly assignment drop threshold.

    Returns:
        List of alert payloads triggered for this week.
    """
    alerts = []
    week_num = curr_week_record.get("week_number", 1)

    # 1. Attendance Drop Check
    curr_att = float(curr_week_record.get("attendance_rate", 100.0))
    prev_att = float(prev_week_record["attendance_rate"]) if prev_week_record else None
    att_alert = check_attendance_drop(
        curr_attendance=curr_att,
        prev_attendance=prev_att,
        floor_threshold=attendance_floor,
        drop_threshold=attendance_drop_threshold,
    )
    if att_alert:
        att_alert["student_id"] = student_id
        att_alert["week_number"] = week_num
        alerts.append(att_alert)

    # 2. Assignment Completion Check
    curr_assign = float(curr_week_record.get("assignment_completion", 100.0))
    prev_assign = float(prev_week_record["assignment_completion"]) if prev_week_record else None
    assign_alert = check_assignment_drop(
        curr_completion=curr_assign,
        prev_completion=prev_assign,
        floor_threshold=assignment_floor,
        drop_threshold=assignment_drop_threshold,
    )
    if assign_alert:
        assign_alert["student_id"] = student_id
        assign_alert["week_number"] = week_num
        alerts.append(assign_alert)

    # 3. Model Probability Jump Check (if present in logs)
    if "at_risk_probability" in curr_week_record and curr_week_record["at_risk_probability"] is not None:
        curr_p = float(curr_week_record["at_risk_probability"])
        prev_p = float(prev_week_record["at_risk_probability"]) if prev_week_record and prev_week_record.get("at_risk_probability") is not None else None
        jump_alert = check_probability_jump(prev_p, curr_p, jump_threshold=prob_jump_threshold)
        if jump_alert:
            jump_alert["student_id"] = student_id
            jump_alert["week_number"] = week_num
            alerts.append(jump_alert)

    # 4. Risk Category Escalation Check (if present in logs)
    if "risk_category" in curr_week_record:
        curr_cat = curr_week_record["risk_category"]
        prev_cat = prev_week_record.get("risk_category") if prev_week_record else None
        esc_alert = check_risk_category_escalation(prev_cat, curr_cat)
        if esc_alert:
            esc_alert["student_id"] = student_id
            esc_alert["week_number"] = week_num
            alerts.append(esc_alert)

    return alerts


def create_alert(
    student_id: str,
    alert_type: str,
    reason: str,
    risk_level: str,
    status: str = "Open",
    alert_date: Optional[datetime] = None,
    data_source: str = "SIMULATED",
    session: Optional[Session] = None,
) -> Alert:
    """Persist a new early warning alert to the database.

    Args:
        student_id: Unique student ID.
        alert_type: Type code (e.g. 'RISK_ESCALATION', 'ATTENDANCE_DROP').
        reason: Plain-text explanation of why the alert triggered.
        risk_level: 'Low', 'Medium', or 'High'.
        status: 'Open' (default), 'Reviewed', or 'Dismissed'.
        alert_date: Timestamp of event. Defaults to now.
        data_source: Provenance flag ('SIMULATED' or 'REAL_UCI').
        session: Active session.

    Returns:
        Created Alert ORM instance.
    """
    if alert_date is None:
        alert_date = _utcnow()

    def _create(s: Session) -> Alert:
        alert = Alert(
            student_id=student_id,
            alert_date=alert_date,
            alert_type=alert_type,
            reason=reason,
            risk_level=risk_level,
            status=status,
            data_source=data_source,
        )
        s.add(alert)
        s.flush()
        return alert

    if session is not None:
        return _create(session)
    with get_db_session() as s:
        return _create(s)


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


def get_open_alerts(
    student_id: Optional[str] = None,
    min_risk_level: Optional[str] = None,
    session: Optional[Session] = None,
) -> List[Alert]:
    """Retrieve all open alerts, optionally filtered by student or minimum risk level.

    Args:
        student_id: Optional student filter.
        min_risk_level: 'Medium' or 'High' filter.
        session: Active session.

    Returns:
        List of matching Alert ORM instances ordered by alert_date descending.
    """
    def _query(s: Session) -> List[Alert]:
        stmt = select(Alert).where(Alert.status == "Open")
        if student_id:
            stmt = stmt.where(Alert.student_id == student_id)

        if min_risk_level:
            min_rank = RISK_LEVEL_RANKS.get(min_risk_level, 1)
            allowed_levels = [lvl for lvl, rk in RISK_LEVEL_RANKS.items() if rk >= min_rank]
            stmt = stmt.where(Alert.risk_level.in_(allowed_levels))

        stmt = stmt.order_by(Alert.alert_date.desc(), Alert.id.desc())
        return list(s.scalars(stmt).all())

    if session is not None:
        return _query(session)
    with get_db_session() as s:
        return _query(s)


def monitor_cohort_weekly_transition(
    cohort_weekly_df: pd.DataFrame,
    session: Optional[Session] = None,
) -> List[Alert]:
    """Scan longitudinal weekly cohort data, generate alerts, and persist them.

    Args:
        cohort_weekly_df: DataFrame containing weekly rows for multiple students.
        session: Active session.

    Returns:
        List of generated Alert ORM instances.
    """
    persisted_alerts = []

    def _scan(s: Session) -> List[Alert]:
        nonlocal persisted_alerts
        # Sort by student_id and week_number
        df_sorted = cohort_weekly_df.sort_values(by=["student_id", "week_number"])

        for st_id, group in df_sorted.groupby("student_id"):
            records = group.to_dict(orient="records")
            for i in range(len(records)):
                curr_rec = records[i]
                prev_rec = records[i - 1] if i > 0 else None

                alerts_triggered = evaluate_student_weekly_monitoring(
                    student_id=str(st_id),
                    curr_week_record=curr_rec,
                    prev_week_record=prev_rec,
                )

                for a_dict in alerts_triggered:
                    alert_obj = create_alert(
                        student_id=a_dict["student_id"],
                        alert_type=a_dict["alert_type"],
                        reason=f"Week {a_dict.get('week_number', curr_rec.get('week_number', 1))}: {a_dict['reason']}",
                        risk_level=a_dict["risk_level"],
                        status="Open",
                        data_source=curr_rec.get("data_source", "SIMULATED"),
                        session=s,
                    )
                    persisted_alerts.append(alert_obj)

        return persisted_alerts

    if session is not None:
        return _scan(session)
    with get_db_session() as s:
        return _scan(s)


def log_new_weekly_performance_and_rescore(
    student_id: str,
    attendance_rate: float,
    quiz_average: float,
    assignment_completion: float,
    study_hours: float,
    notes: Optional[str] = None,
    trajectory_type: Optional[str] = None,
    data_source: str = "SIMULATED",
    session: Optional[Session] = None,
) -> Dict[str, Any]:
    """Record a new week of performance logs, re-score student risk, and trigger early-warning alerts.

    Args:
        student_id: Unique student ID.
        attendance_rate: Weekly attendance percentage (0.0 - 100.0).
        quiz_average: Weekly quiz mark (0.0 - 20.0).
        assignment_completion: Weekly assignment completion percentage (0.0 - 100.0).
        study_hours: Total self-study hours logged during the week.
        notes: Optional teacher/counselor observation notes.
        trajectory_type: Optional label ('STABLE', 'DECLINING', 'IMPROVING').
        data_source: Provenance flag ('SIMULATED' or 'REAL').
        session: Active database session.

    Returns:
        Summary report dictionary with new score, risk probability, risk category,
        and list of newly generated alert ORM objects or dictionaries.
    """
    from src.classification import load_classification_model, predict_risk
    from src.features import add_engineered_features
    from src.regression import load_model, predict_score

    def _execute(s: Session) -> Dict[str, Any]:
        student = s.scalar(select(Student).where(Student.student_id == student_id))
        if not student:
            raise ValueError(f"Student with ID '{student_id}' does not exist.")

        acad = s.scalar(select(AcademicRecord).where(AcademicRecord.student_id == student_id))

        # Query existing weekly records
        existing_stmt = (
            select(WeeklyPerformance)
            .where(WeeklyPerformance.student_id == student_id)
            .order_by(WeeklyPerformance.week_number)
        )
        existing_weeks = list(s.scalars(existing_stmt).all())
        next_week = (existing_weeks[-1].week_number + 1) if existing_weeks else 1
        prev_week_obj = existing_weeks[-1] if existing_weeks else None

        # Fetch current prediction before rescoring (for probability jump tracking)
        prev_pred = s.scalar(
            select(Prediction).where(
                Prediction.student_id == student_id,
                Prediction.scenario == "Scenario B (Mid-Semester, With G1/G2)",
            )
        )
        prev_prob = prev_pred.at_risk_probability if prev_pred else None
        prev_tier = prev_pred.risk_category if prev_pred else None

        # 1. Insert new weekly performance row
        new_wp = WeeklyPerformance(
            student_id=student_id,
            week_number=next_week,
            attendance_rate=attendance_rate,
            quiz_average=quiz_average,
            assignment_completion=assignment_completion,
            study_hours=study_hours,
            trajectory_type=trajectory_type,
            notes=notes,
            data_source=data_source,
        )
        s.add(new_wp)
        s.flush()

        # 2. Derive updated features for ML inference
        # Study hours mapped to studytime category (1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h)
        if study_hours < 2.5:
            st_cat = 1
        elif study_hours <= 5.0:
            st_cat = 2
        elif study_hours <= 10.0:
            st_cat = 3
        else:
            st_cat = 4

        current_absences = int(acad.absences) if (acad and acad.absences is not None) else 4
        if attendance_rate < 80.0:
            current_absences += 1

        prior_g1 = float(acad.G1) if (acad and acad.G1 is not None) else 10.0
        prior_g2 = float(acad.G2) if (acad and acad.G2 is not None) else 10.0
        # Blend quiz average into mid-term performance
        updated_g2 = 0.7 * prior_g2 + 0.3 * quiz_average

        student_row = {
            "school": student.school or "GP",
            "subject": acad.subject if acad else "math",
            "reason": student.reason or "course",
            "traveltime": acad.traveltime if acad else 1,
            "studytime": st_cat,
            "failures": acad.failures if acad else 0,
            "schoolsup": acad.schoolsup if acad else "no",
            "famsup": acad.famsup if acad else "no",
            "paid": acad.paid if acad else "no",
            "activities": acad.activities if acad else "yes",
            "nursery": acad.nursery if acad else "yes",
            "higher": acad.higher if acad else "yes",
            "internet": acad.internet if acad else "yes",
            "romantic": acad.romantic if acad else "no",
            "famrel": acad.famrel if acad else 4,
            "freetime": acad.freetime if acad else 3,
            "goout": acad.goout if acad else 3,
            "Dalc": acad.Dalc if acad else 1,
            "Walc": acad.Walc if acad else 2,
            "health": acad.health if acad else 3,
            "absences": current_absences,
            "G1": prior_g1,
            "G2": updated_g2,
        }

        # 3. Model Re-scoring
        clf, meta_clf = load_classification_model("B")
        reg, meta_reg = load_model("B")

        input_df = add_engineered_features(pd.DataFrame([student_row]), include_trend=True)
        risk_res = predict_risk(clf, meta_clf, input_df, threshold=0.15)
        score_res = predict_score(reg, meta_reg, input_df)

        new_prob = float(risk_res["probabilities"][0])
        new_tier = str(risk_res["risk_tiers"][0])
        new_score = float(score_res[0])

        # 4. Update or insert Prediction record
        if prev_pred:
            prev_pred.predicted_score = new_score
            prev_pred.at_risk_probability = new_prob
            prev_pred.is_at_risk = 1 if new_prob >= 0.15 else 0
            prev_pred.risk_category = new_tier
            prev_pred.model_version = meta_clf.get("model_name", "Random Forest")
            prev_pred.prediction_date = _utcnow()
        else:
            new_pred_obj = Prediction(
                student_id=student_id,
                scenario="Scenario B (Mid-Semester, With G1/G2)",
                model_version=meta_clf.get("model_name", "Random Forest"),
                predicted_score=new_score,
                at_risk_probability=new_prob,
                is_at_risk=1 if new_prob >= 0.15 else 0,
                risk_category=new_tier,
                data_source=data_source,
                prediction_date=_utcnow(),
            )
            s.add(new_pred_obj)
        s.flush()

        # 5. Evaluate Early-Warning Alert Rules
        curr_week_dict = {
            "week_number": next_week,
            "attendance_rate": attendance_rate,
            "quiz_average": quiz_average,
            "assignment_completion": assignment_completion,
            "study_hours": study_hours,
            "at_risk_probability": new_prob,
            "risk_category": new_tier,
        }
        prev_week_dict = None
        if prev_week_obj:
            prev_week_dict = {
                "week_number": prev_week_obj.week_number,
                "attendance_rate": prev_week_obj.attendance_rate,
                "quiz_average": prev_week_obj.quiz_average,
                "assignment_completion": prev_week_obj.assignment_completion,
                "study_hours": prev_week_obj.study_hours,
                "at_risk_probability": prev_prob,
                "risk_category": prev_tier,
            }

        alerts_triggered = evaluate_student_weekly_monitoring(
            student_id=student_id,
            curr_week_record=curr_week_dict,
            prev_week_record=prev_week_dict,
        )

        persisted_alerts = []
        for a in alerts_triggered:
            al_obj = create_alert(
                student_id=student_id,
                alert_type=a["alert_type"],
                reason=f"Week {next_week}: {a['reason']}",
                risk_level=a["risk_level"],
                status="Open",
                data_source=data_source,
                session=s,
            )
            persisted_alerts.append(al_obj)

        return {
            "status": "success",
            "student_id": student_id,
            "week_number": next_week,
            "new_predicted_score": new_score,
            "new_risk_probability": new_prob,
            "new_risk_tier": new_tier,
            "alerts_count": len(persisted_alerts),
            "alerts": [
                {
                    "id": al.id,
                    "alert_type": al.alert_type,
                    "risk_level": al.risk_level,
                    "reason": al.reason,
                }
                for al in persisted_alerts
            ],
        }

    if session is not None:
        return _execute(session)
    with get_db_session() as s:
        return _execute(s)


if __name__ == "__main__":
    print("Testing alert generation functions...")
    esc = check_risk_category_escalation("Low", "High")
    print("Escalation alert:", esc)

    jump = check_probability_jump(0.20, 0.45, 0.15)
    print("Probability jump alert:", jump)

    att = check_attendance_drop(55.0, 85.0)
    print("Attendance drop alert:", att)

    assign = check_assignment_drop(40.0, 75.0)
    print("Assignment drop alert:", assign)
    print("Monitoring module verified.")
