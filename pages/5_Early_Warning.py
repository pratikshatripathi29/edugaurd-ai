"""
EduGuard AI — Longitudinal Early-Warning Monitoring & Intervention Tracker (Stage 10)

This module provides comprehensive operational tracking for academic advisors:
1. Active alerts table with "Mark Reviewed" resolution actions (persisted to SQLite).
2. Longitudinal trend charts highlighting students with rising risk or declining trajectories (labeled SIMULATED).
3. Complete Intervention Tracker:
   - Form to log new interventions with automatically captured baseline risk, score, and attendance.
   - Follow-up evaluation form to record post-intervention metrics, computing before/after deltas and "Observed change".
   - Mandatory causal disclaimer: "Observed improvement is not proof that the intervention caused it."
4. Live weekly performance logging: allows counselors to input a new week of data (Week 13+),
   automatically re-score the student using champion ML models, and trigger real-time alerts.
"""

from datetime import datetime, timezone
from pathlib import Path
import sys

# Ensure project root is on sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import desc, func, select
import streamlit as st

from src.config import DATABASE_PATH, RISK_PASS_MARK, RISK_THRESHOLDS
from src.database import (
    AcademicRecord,
    Alert,
    Intervention,
    Prediction,
    Student,
    WeeklyPerformance,
    add_intervention,
    ensure_database_ready,
    get_all_interventions,
    get_db_session,
    record_intervention_followup,
)
from src.monitoring import (
    get_open_alerts,
    log_new_weekly_performance_and_rescore,
    update_alert_status,
)
from src.utils import (
    apply_plotly_theme,
    get_advisory_notice_html,
    get_kpi_html,
    get_risk_badge_html,
    get_simulated_tag_html,
    load_custom_css,
)

# Page configuration
st.set_page_config(
    page_title="EduGuard AI | Early-Warning & Interventions",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Ensure database tables exist and are seeded before any queries run
ensure_database_ready()

load_custom_css()

# ==============================================================================
# 1. Header & Governance Disclaimer Banner
# ==============================================================================
st.markdown(
    """
    <div style='margin-bottom: 16px;'>
        <h2 style='color: #0f172a; font-weight: 800; margin-bottom: 4px;'>
            🚨 Longitudinal Early-Warning & Intervention Tracker
        </h2>
        <p style='color: #64748b; font-size: 0.95rem; margin: 0;'>
            Track trajectory shifts, resolve counselor alerts, log intervention follow-ups, and enter live weekly updates.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(get_advisory_notice_html(), unsafe_allow_html=True)


# ==============================================================================
# 2. Database Retrieval Helpers (Cached)
# ==============================================================================
@st.cache_data(ttl=15)
def get_monitoring_overview():
    """Fetch high-level telemetry for alerts and interventions."""
    if not DATABASE_PATH.exists():
        return {}

    with get_db_session() as session:
        total_alerts = session.scalar(select(func.count(Alert.id))) or 0
        open_alerts_count = session.scalar(select(func.count(Alert.id)).where(Alert.status == "Open")) or 0
        high_risk_alerts = session.scalar(
            select(func.count(Alert.id)).where(Alert.status == "Open", Alert.risk_level == "High")
        ) or 0
        total_interventions = session.scalar(select(func.count(Intervention.id))) or 0
        completed_interventions = session.scalar(
            select(func.count(Intervention.id)).where(Intervention.status == "Completed")
        ) or 0

        # Unique students with declining weekly trajectories
        declining_students = list(
            session.scalars(
                select(WeeklyPerformance.student_id)
                .where(WeeklyPerformance.trajectory_type == "DECLINING")
                .distinct()
            ).all()
        )

        return {
            "total_alerts": total_alerts,
            "open_alerts": open_alerts_count,
            "high_risk_alerts": high_risk_alerts,
            "total_interventions": total_interventions,
            "completed_interventions": completed_interventions,
            "declining_students_count": len(declining_students),
            "declining_student_ids": declining_students,
        }


@st.cache_data(ttl=15)
def get_all_student_ids_list():
    """Retrieve list of all student IDs."""
    if not DATABASE_PATH.exists():
        return []
    with get_db_session() as session:
        return list(session.scalars(select(Student.student_id).order_by(Student.student_id)).all())


overview = get_monitoring_overview()

# Telemetry KPI row
col_k1, col_k2, col_k3, col_k4 = st.columns(4)
with col_k1:
    st.markdown(
        get_kpi_html("Active Open Alerts", f"{overview.get('open_alerts', 0):,}", f"{overview.get('high_risk_alerts', 0)} High Priority", border_color="#ef4444"),
        unsafe_allow_html=True,
    )
with col_k2:
    st.markdown(
        get_kpi_html("Declining Trajectories", f"{overview.get('declining_students_count', 0)} Students", "Sudden midterm drop", border_color="#f59e0b"),
        unsafe_allow_html=True,
    )
with col_k3:
    st.markdown(
        get_kpi_html("Recorded Interventions", f"{overview.get('total_interventions', 0):,}", f"{overview.get('completed_interventions', 0)} Completed Follow-ups", border_color="#0d9488"),
        unsafe_allow_html=True,
    )
with col_k4:
    st.markdown(
        get_kpi_html("Data Provenance", "SIMULATED", "Weekly logs & trajectories", border_color="#8b5cf6"),
        unsafe_allow_html=True,
    )

st.markdown("<hr style='margin: 18px 0;'>", unsafe_allow_html=True)


# ==============================================================================
# 3. Main Operational Tabs
# ==============================================================================
tab_alerts, tab_trends, tab_interventions, tab_log_week = st.tabs([
    "🚨 Active Alerts Queue & Triage",
    f"📈 Rising Risk & Weekly Trends {get_simulated_tag_html()}",
    "📋 Intervention Tracker & Outcomes",
    "➕ Log New Week & Auto-Rescore",
])


# ==============================================================================
# Tab 1: Active Alerts Queue & Triage
# ==============================================================================
with tab_alerts:
    st.markdown("<div class='section-header'>🚨 Counselor Alerts Queue & Rapid Triage</div>", unsafe_allow_html=True)

    col_flt1, col_flt2 = st.columns([1, 1])
    with col_flt1:
        status_filter = st.radio("Filter by Alert Status:", ["Open", "Reviewed", "All"], horizontal=True, index=0)
    with col_flt2:
        risk_filter = st.selectbox("Minimum Risk Level:", ["All Risk Levels", "High", "Medium", "Low"], index=0)

    # Fetch alerts from database
    with get_db_session() as session:
        stmt = select(Alert)
        if status_filter != "All":
            stmt = stmt.where(Alert.status == status_filter)
        if risk_filter != "All Risk Levels":
            stmt = stmt.where(Alert.risk_level == risk_filter)
        stmt = stmt.order_by(Alert.alert_date.desc(), Alert.id.desc())
        alerts_list = list(session.scalars(stmt).all())

    if alerts_list:
        st.caption(f"Displaying {len(alerts_list)} alerts matching selected criteria.")

        for alert_item in alerts_list[:30]:  # Display top 30
            c_card, c_act = st.columns([4, 1])

            with c_card:
                dt_str = alert_item.alert_date.strftime("%Y-%m-%d %H:%M") if alert_item.alert_date else "N/A"
                border_color = "#ef4444" if alert_item.risk_level == "High" else "#f59e0b" if alert_item.risk_level == "Medium" else "#0d9488"
                status_badge = f"<span style='color: {'#10b981' if alert_item.status == 'Reviewed' else '#ef4444'}; font-weight: 700;'>[{alert_item.status.upper()}]</span>"

                st.markdown(
                    f"""
                    <div style='background-color: #ffffff; border-left: 5px solid {border_color}; border-top: 1px solid #e2e8f0; border-right: 1px solid #e2e8f0; border-bottom: 1px solid #e2e8f0; border-radius: 0 8px 8px 0; padding: 12px 16px; margin-bottom: 8px;'>
                        <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;'>
                            <span style='font-weight: 800; color: #0f172a; font-size: 0.92rem;'>
                                #{alert_item.id} • Student: {alert_item.student_id} • {alert_item.alert_type}
                            </span>
                            <div>
                                {get_risk_badge_html(alert_item.risk_level)}
                                {status_badge}
                                <span style='font-size: 0.78rem; color: #64748b; margin-left: 8px;'>{dt_str}</span>
                            </div>
                        </div>
                        <div style='color: #475569; font-size: 0.86rem; line-height: 1.4;'>
                            {alert_item.reason}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with c_act:
                if alert_item.status == "Open":
                    if st.button("Mark Reviewed", key=f"btn_rev_{alert_item.id}", use_container_width=True):
                        update_alert_status(alert_item.id, "Reviewed")
                        st.toast(f"Alert #{alert_item.id} for {alert_item.student_id} marked as Reviewed!")
                        st.rerun()
                else:
                    st.caption(f"Resolved {alert_item.resolved_at.strftime('%m-%d') if alert_item.resolved_at else 'Reviewed'}")
    else:
        st.info("No alerts found matching the active filter criteria.")


# ==============================================================================
# Tab 2: Rising Risk & Longitudinal Trends [SIMULATED]
# ==============================================================================
with tab_trends:
    st.markdown(
        f"<div class='section-header'>📈 Longitudinal Trajectories & Rising Risk Detection {get_simulated_tag_html()}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div style='background-color: #f8fafc; border-left: 4px solid #8b5cf6; padding: 10px 16px; border-radius: 0 8px 8px 0; margin-bottom: 16px;'>
            <span style='font-weight: 700; color: #4c1d95;'>⚠️ Data Provenance Notice:</span>
            <span style='color: #334155; font-size: 0.88rem;'>
                All weekly attendance, quiz marks, and assignment completion logs are <strong>SIMULATED</strong>
                anchored to real baseline attributes. This data powers early-warning pattern detection.
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    all_students = get_all_student_ids_list()
    declining_ids = overview.get("declining_student_ids", [])

    col_tr1, col_tr2 = st.columns([2, 1])
    with col_tr1:
        # Pre-select first declining student if available
        default_idx = 0
        if declining_ids and declining_ids[0] in all_students:
            default_idx = all_students.index(declining_ids[0])

        selected_trend_student = st.selectbox(
            "Select Student to Inspect Weekly Trajectory:",
            options=all_students,
            index=default_idx,
            help="Select a student to display their weekly performance metrics and trajectory classification.",
        )

    with col_tr2:
        is_declining = selected_trend_student in declining_ids
        badge_text = "🚨 Declining Trajectory" if is_declining else "✅ Stable Trajectory"
        badge_color = "#ef4444" if is_declining else "#10b981"
        st.markdown(
            f"""
            <div style='margin-top: 26px; padding: 10px 16px; background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px;'>
                <span style='font-size: 0.82rem; color: #64748b; font-weight: 600;'>Trajectory Type:</span><br>
                <strong style='color: {badge_color}; font-size: 1.0rem;'>{badge_text}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Fetch weekly logs for selected student
    with get_db_session() as session:
        stmt_w = (
            select(WeeklyPerformance)
            .where(WeeklyPerformance.student_id == selected_trend_student)
            .order_by(WeeklyPerformance.week_number)
        )
        wp_rows = list(session.scalars(stmt_w).all())

    if wp_rows:
        df_weekly = pd.DataFrame([
            {
                "Week": r.week_number,
                "Attendance Rate (%)": r.attendance_rate,
                "Quiz Average (0-20)": r.quiz_average,
                "Assignment Completion (%)": r.assignment_completion,
                "Study Hours": r.study_hours,
                "Trajectory": r.trajectory_type or "STABLE",
                "Data Source": r.data_source,
            }
            for r in wp_rows
        ])

        fig_trend = go.Figure()

        # Attendance Rate Line
        fig_trend.add_trace(
            go.Scatter(
                x=df_weekly["Week"],
                y=df_weekly["Attendance Rate (%)"],
                name="Attendance Rate (%)",
                line=dict(color="#0d9488", width=3),
                mode="lines+markers",
            )
        )

        # Assignment Completion Line
        fig_trend.add_trace(
            go.Scatter(
                x=df_weekly["Week"],
                y=df_weekly["Assignment Completion (%)"],
                name="Assignment Completion (%)",
                line=dict(color="#6366f1", width=2, dash="dash"),
                mode="lines+markers",
            )
        )

        # Quiz Average (Scaled to 100% for dual-axis visual alignment)
        fig_trend.add_trace(
            go.Scatter(
                x=df_weekly["Week"],
                y=df_weekly["Quiz Average (0-20)"] * 5.0,
                name="Quiz Average (Scaled %)",
                line=dict(color="#f59e0b", width=2.5),
                mode="lines+markers",
            )
        )

        # Critical attendance floor reference
        fig_trend.add_hline(y=75.0, line_dash="dot", line_color="#ef4444", annotation_text="Attendance Floor (75%)")

        apply_plotly_theme(fig_trend, f"Longitudinal Weekly Tracking for {selected_trend_student} [SIMULATED]")
        fig_trend.update_xaxes(title_text="Academic Week Number", dtick=1)
        fig_trend.update_yaxes(title_text="Percentage (%) / Scaled Quiz Score", range=[0, 105])
        st.plotly_chart(fig_trend, use_container_width=True)

        st.dataframe(df_weekly, use_container_width=True, hide_index=True)
    else:
        st.info(f"No weekly records logged for {selected_trend_student}.")


# ==============================================================================
# Tab 3: Intervention Tracker & Outcomes
# ==============================================================================
with tab_interventions:
    st.markdown("<div class='section-header'>📋 Pedagogical Intervention Tracker & Outcome Evaluation</div>", unsafe_allow_html=True)

    # Mandatory Ethical Causal Notice
    st.markdown(
        """
        <div style='background-color: #fffbeb; border-left: 5px solid #f59e0b; padding: 14px 18px; border-radius: 0 8px 8px 0; margin-bottom: 20px;'>
            <div style='font-weight: 700; color: #92400e; font-size: 0.90rem; margin-bottom: 4px;'>
                ⚠️ Critical Evaluation Notice: Correlation vs. Causal Impact
            </div>
            <p style='color: #78350f; font-size: 0.86rem; line-height: 1.5; margin: 0;'>
                <strong>Observed improvement is not proof that the intervention caused it.</strong>
                Academic trajectory shifts reflect multiple converging factors including personal effort, peer support,
                natural syllabus variation, external tutoring, and family dynamics. This tracker provides structured
                observational documentation, not randomized causal proof.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_int_form, col_int_followup = st.columns([1, 1], gap="large")

    # Form A: Record New Intervention with Baseline Snapshot
    with col_int_form:
        st.markdown("<b>➕ Record New Intervention Plan</b>", unsafe_allow_html=True)
        st.caption("Captures baseline ML risk probability, predicted mark, and attendance automatically.")

        with st.form("form_new_intervention"):
            st_target = st.selectbox("Select Student:", options=all_students, index=0)

            # Auto-retrieve student's current baseline
            with get_db_session() as s:
                acad_rec = s.scalar(select(AcademicRecord).where(AcademicRecord.student_id == st_target))
                pred_rec = s.scalar(
                    select(Prediction).where(
                        Prediction.student_id == st_target,
                        Prediction.scenario == "Scenario B (Mid-Semester, With G1/G2)",
                    )
                )

            base_prob = float(pred_rec.at_risk_probability) if pred_rec and pred_rec.at_risk_probability is not None else 0.50
            base_score = float(pred_rec.predicted_score) if pred_rec and pred_rec.predicted_score is not None else 10.0
            base_att = 90.0 - (float(acad_rec.absences) * 1.5) if acad_rec and acad_rec.absences else 85.0

            st.info(f"Baseline Snapshot for {st_target}: Risk: **{base_prob * 100:.1f}%** | Score: **{base_score:.2f} / 20** | Attendance: **{base_att:.1f}%**")

            i_type = st.selectbox(
                "Intervention Type:",
                ["ACADEMIC_REMEDIAL_PLAN", "MANDATORY_TUTORING", "ATTENDANCE_COUNSELING", "PEER_STUDY_GROUP", "FACULTY_OFFICE_HOURS", "WELLNESS_SUPPORT"],
            )
            i_advisor = st.text_input("Assigned Educator / Counselor:", value="Academic Success Advisor")
            i_desc = st.text_area("Intervention Strategy & Teacher Notes:", height=100, placeholder="Document agreed action steps, weekly study goals, and referral services...")
            submit_new_i = st.form_submit_button("💾 Commit New Intervention", type="primary", use_container_width=True)

            if submit_new_i:
                if not i_desc.strip():
                    st.error("Please enter intervention notes or strategy before saving.")
                else:
                    new_item = add_intervention(
                        student_id=st_target,
                        intervention_type=i_type,
                        description=i_desc,
                        assigned_to=i_advisor,
                        status="Scheduled",
                        baseline_risk_prob=base_prob,
                        baseline_predicted_score=base_score,
                        baseline_attendance=base_att,
                        data_source="TEACHER_LOGGED",
                    )
                    st.success(f"Intervention #{new_item.id} logged successfully for {st_target}!")
                    st.rerun()

    # Form B: Record Follow-Up Review & Compute Changes
    with col_int_followup:
        st.markdown("<b>📝 Record Follow-Up Evaluation & Compute Deltas</b>", unsafe_allow_html=True)
        st.caption("Compare baseline diagnostic metrics against observed follow-up attainment.")

        # Query existing scheduled or in-progress interventions
        with get_db_session() as s:
            all_int_items = list(s.scalars(select(Intervention).order_by(desc(Intervention.id))).all())

        if all_int_items:
            int_options = [
                f"#{i.id} - {i.student_id} ({i.intervention_type}) [{i.status}]"
                for i in all_int_items
            ]
            selected_int_str = st.selectbox("Select Existing Intervention to Follow-Up:", options=int_options)
            sel_id = int(selected_int_str.split(" - ")[0].replace("#", ""))

            # Get target item
            target_interv = next(i for i in all_int_items if i.id == sel_id)

            st.write(
                f"**Baseline Indicators:** Risk: `{target_interv.baseline_risk_prob * 100:.1f}%`"
                if target_interv.baseline_risk_prob is not None
                else "**Baseline Indicators:** Not explicitly captured"
            )
            if target_interv.baseline_predicted_score is not None:
                st.write(f"**Baseline Score:** `{target_interv.baseline_predicted_score:.2f} / 20` | **Baseline Attendance:** `{target_interv.baseline_attendance:.1f}%`")

            with st.form("form_followup"):
                f_att = st.slider("Follow-Up Attendance Rate (%):", 0.0, 100.0, float(target_interv.followup_attendance or target_interv.baseline_attendance or 85.0), step=1.0)
                f_marks = st.slider("Follow-Up Exam Mark / Quiz Average (0-20):", 0.0, 20.0, float(target_interv.followup_marks or target_interv.baseline_predicted_score or 10.0), step=0.5)
                f_risk = st.slider("Follow-Up Re-Scored At-Risk Probability (%):", 0.0, 100.0, float((target_interv.followup_risk_prob or target_interv.baseline_risk_prob or 0.5) * 100.0), step=1.0) / 100.0
                f_status = st.selectbox("Updated Status:", ["Completed", "In Progress", "Scheduled", "Cancelled"], index=0)
                f_notes = st.text_area("Follow-Up Clinical / Counseling Notes:", value=target_interv.followup_notes or "", height=80, placeholder="Document observed behavioral changes, student response, and ongoing needs...")

                submit_followup = st.form_submit_button("✅ Record Follow-Up Review", type="primary", use_container_width=True)

                if submit_followup:
                    updated_interv = record_intervention_followup(
                        intervention_id=target_interv.id,
                        followup_attendance=f_att,
                        followup_marks=f_marks,
                        followup_risk_prob=f_risk,
                        followup_notes=f_notes,
                        status=f_status,
                    )
                    st.success(f"Follow-up logged for Intervention #{target_interv.id}! Observed Change: **{updated_interv.observed_change}**")
                    st.rerun()
        else:
            st.info("No recorded interventions available for follow-up.")

    # Historical Interventions Ledger Table
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<b>All Recorded Interventions & Outcome Shifts:</b>", unsafe_allow_html=True)

    interventions_all = get_all_interventions()
    if interventions_all:
        rows_ledger = []
        for it in interventions_all:
            base_p = f"{it.baseline_risk_prob * 100:.1f}%" if it.baseline_risk_prob is not None else "N/A"
            f_p = f"{it.followup_risk_prob * 100:.1f}%" if it.followup_risk_prob is not None else "Pending"
            base_s = f"{it.baseline_predicted_score:.1f}" if it.baseline_predicted_score is not None else "N/A"
            f_s = f"{it.followup_marks:.1f}" if it.followup_marks is not None else "Pending"

            change_badge = it.observed_change or "Pending Review"

            rows_ledger.append({
                "Ticket": f"#{it.id}",
                "Student": it.student_id,
                "Type": it.intervention_type,
                "Status": it.status,
                "Date Logged": it.date_logged.strftime("%Y-%m-%d") if it.date_logged else "N/A",
                "Baseline Risk": base_p,
                "Follow-Up Risk": f_p,
                "Baseline Score": base_s,
                "Follow-Up Score": f_s,
                "Observed Change": change_badge,
                "Counselor": it.assigned_to,
            })
        st.dataframe(pd.DataFrame(rows_ledger), use_container_width=True, hide_index=True)
    else:
        st.info("No intervention records in the database.")


# ==============================================================================
# Tab 4: Log New Week & Auto-Rescore
# ==============================================================================
with tab_log_week:
    st.markdown("<div class='section-header'>➕ Log New Academic Week & Live Re-Score</div>", unsafe_allow_html=True)
    st.caption(
        "Enter live or simulated weekly observation data (Week 13+) for a student. The system commits "
        "the record to the database, automatically re-scores risk via the champion ML pipeline, "
        "and checks for early-warning alert triggers in real time."
    )

    col_nw1, col_nw2 = st.columns([1, 1], gap="large")

    with col_nw1:
        st_select_log = st.selectbox("Select Student for Weekly Update:", options=all_students, key="sb_log_week")

        # Find existing weeks
        with get_db_session() as s:
            weeks_logged = list(
                s.scalars(
                    select(WeeklyPerformance.week_number)
                    .where(WeeklyPerformance.student_id == st_select_log)
                    .order_by(WeeklyPerformance.week_number)
                ).all()
            )

        next_w_num = (weeks_logged[-1] + 1) if weeks_logged else 1
        st.info(f"Student **{st_select_log}** currently has **{len(weeks_logged)} weeks** logged. Next Entry: **Week {next_w_num}**.")

        with st.form("form_log_new_week"):
            in_att = st.slider("Week Attendance Rate (%):", 0.0, 100.0, 85.0, step=1.0, help="Percentage of lectures/labs attended.")
            in_quiz = st.slider("Week Quiz / Assessment Average (0-20):", 0.0, 20.0, 10.5, step=0.5, help="Weekly quiz result on 0-20 scale.")
            in_assign = st.slider("Assignment Completion Rate (%):", 0.0, 100.0, 80.0, step=5.0, help="Fraction of homework delivered.")
            in_hours = st.slider("Weekly Self-Directed Study Hours:", 0.0, 30.0, 5.0, step=0.5, help="Independent study and homework hours.")
            in_notes = st.text_input("Teacher / Proctor Observations:", placeholder="e.g. Returned after illness, showed improved focus...")
            in_traj = st.selectbox("Trajectory Classification:", ["STABLE", "DECLINING", "IMPROVING"], index=0)

            submit_week = st.form_submit_button("🚀 Submit Week & Re-Score Student", type="primary", use_container_width=True)

    with col_nw2:
        st.markdown("<b>⚡ Live Re-Scoring Results & Triggered Alerts</b>", unsafe_allow_html=True)

        if submit_week:
            with st.spinner(f"Committing Week {next_w_num} and re-scoring with ML pipeline..."):
                try:
                    res_dict = log_new_weekly_performance_and_rescore(
                        student_id=st_select_log,
                        attendance_rate=in_att,
                        quiz_average=in_quiz,
                        assignment_completion=in_assign,
                        study_hours=in_hours,
                        notes=in_notes,
                        trajectory_type=in_traj,
                        data_source="SIMULATED",
                    )

                    st.success(f"✅ Week {res_dict['week_number']} successfully committed for student {st_select_log}!")

                    # Results KPIs
                    r_col1, r_col2, r_col3 = st.columns(3)
                    with r_col1:
                        st.metric("New Predicted G3", f"{res_dict['new_predicted_score']:.2f} / 20")
                    with r_col2:
                        st.metric("New Risk Probability", f"{res_dict['new_risk_probability'] * 100:.1f}%")
                    with r_col3:
                        st.markdown(f"**Risk Tier:**<br>{get_risk_badge_html(res_dict['new_risk_tier'])}", unsafe_allow_html=True)

                    # Triggered Alerts Feedback
                    st.markdown("<br><b>Early-Warning Alerts Generated:</b>", unsafe_allow_html=True)
                    if res_dict["alerts_count"] > 0:
                        for al in res_dict["alerts"]:
                            st.error(f"🚨 **[{al['risk_level']} ALERT]** {al['alert_type']}: {al['reason']}")
                    else:
                        st.info("No distress thresholds breached. Student remains in stable monitoring status.")

                except Exception as err:
                    st.error(f"Failed to record week: {str(err)}")
        else:
            st.info("Fill out the form on the left and submit to view live re-scoring and automated alerts.")

