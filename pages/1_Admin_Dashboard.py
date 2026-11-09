"""
EduGuard AI — Admin & Cohort Dashboard (Stage 7)

Provides institutional decision-support:
1. Live KPI metrics computed directly from database tables.
2. Interactive Plotly charts (Risk distribution, Grade curves, Absences analysis, Subject breakdown).
3. Longitudinal weekly cohort trend (explicitly marked SIMULATED).
4. High-risk student triage queue.
5. Live alerts stream with status review toggles.
6. Data Management: Bulk CSV uploader and Add/Update student forms.
"""

from pathlib import Path
import sys

# Ensure project root is on sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import func, select
import streamlit as st

from src.config import DATABASE_PATH, RISK_PASS_MARK
from src.database import (
    AcademicRecord,
    Alert,
    Prediction,
    Student,
    WeeklyPerformance,
    add_student,
    bulk_import_csv,
    get_db_session,
    update_alert_status,
    update_student,
)
from src.utils import (
    apply_plotly_theme,
    get_advisory_notice_html,
    get_kpi_html,
    get_risk_badge_html,
    get_simulated_tag_html,
    load_custom_css,
)

# Page Configuration
st.set_page_config(
    page_title="EduGuard AI | Admin Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_custom_css()

# ==============================================================================
# 1. Header & Context Banner
# ==============================================================================
st.markdown(
    """
    <div style='margin-bottom: 16px;'>
        <h2 style='color: #0f172a; font-weight: 800; margin-bottom: 4px;'>
            📈 Admin & Cohort Early-Warning Dashboard
        </h2>
        <p style='color: #64748b; font-size: 0.95rem; margin: 0;'>
            Real-time cohort monitoring, risk stratification, and academic intervention triage.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(get_advisory_notice_html(), unsafe_allow_html=True)


# ==============================================================================
# 2. Cached Data Loading Routines
# ==============================================================================
@st.cache_data(ttl=30)
def load_cohort_dataset(scenario: str = "Scenario B") -> pd.DataFrame:
    """Fetch joined student demographic, academic, and model prediction records."""
    if not DATABASE_PATH.exists():
        return pd.DataFrame()

    with get_db_session() as session:
        stmt = (
            select(
                Student.student_id,
                Student.school,
                Student.sex,
                Student.age,
                Student.address,
                AcademicRecord.subject,
                AcademicRecord.studytime,
                AcademicRecord.failures,
                AcademicRecord.absences,
                AcademicRecord.G1,
                AcademicRecord.G2,
                AcademicRecord.G3,
                Prediction.predicted_score,
                Prediction.at_risk_probability,
                Prediction.risk_category,
                Prediction.is_at_risk,
                Prediction.scenario,
            )
            .join(AcademicRecord, Student.student_id == AcademicRecord.student_id)
            .join(Prediction, Student.student_id == Prediction.student_id)
            .where(Prediction.scenario == scenario)
        )
        df = pd.read_sql(stmt, session.bind)
    return df


@st.cache_data(ttl=30)
def load_weekly_cohort_trends() -> pd.DataFrame:
    """Fetch aggregate weekly performance across all 12 simulated weeks."""
    if not DATABASE_PATH.exists():
        return pd.DataFrame()

    with get_db_session() as session:
        stmt = (
            select(
                WeeklyPerformance.week_number,
                func.avg(WeeklyPerformance.attendance_rate).label("avg_attendance"),
                func.avg(WeeklyPerformance.quiz_average).label("avg_quiz"),
                func.avg(WeeklyPerformance.assignment_completion).label("avg_assignment"),
                func.avg(WeeklyPerformance.study_hours).label("avg_hours"),
            )
            .group_by(WeeklyPerformance.week_number)
            .order_by(WeeklyPerformance.week_number)
        )
        df = pd.read_sql(stmt, session.bind)
    return df


@st.cache_data(ttl=30)
def load_alerts_stream(limit: int = 30) -> pd.DataFrame:
    """Fetch recent early-warning alerts."""
    if not DATABASE_PATH.exists():
        return pd.DataFrame()

    with get_db_session() as session:
        stmt = (
            select(
                Alert.id,
                Alert.student_id,
                Alert.alert_type,
                Alert.reason,
                Alert.risk_level,
                Alert.status,
                Alert.alert_date,
                Alert.data_source,
            )
            .order_by(Alert.id.desc())
            .limit(limit)
        )
        df = pd.read_sql(stmt, session.bind)
    return df


# Scenario Selection Toggle
col_head1, col_head2 = st.columns([3, 1])
with col_head2:
    selected_scenario = st.selectbox(
        "Modeling Scenario",
        options=["Scenario B", "Scenario A"],
        help="Scenario A = Early-Warning (No G1/G2). Scenario B = Mid-Semester Update (With G1/G2).",
    )

with st.spinner("Loading cohort metrics from database..."):
    df_cohort = load_cohort_dataset(selected_scenario)
    df_weekly = load_weekly_cohort_trends()
    df_alerts = load_alerts_stream()

if df_cohort.empty:
    st.error("⚠️ No student records found in the database. Please run `python scripts/seed_db.py` to seed records.")
    st.stop()


# ==============================================================================
# 3. Dynamic Real-Time KPI Cards (All Computed Run-Time)
# ==============================================================================
total_students = len(df_cohort)
high_risk_count = len(df_cohort[df_cohort["risk_category"] == "High"])
medium_risk_count = len(df_cohort[df_cohort["risk_category"] == "Medium"])
low_risk_count = len(df_cohort[df_cohort["risk_category"] == "Low"])

high_pct = (high_risk_count / total_students) * 100 if total_students > 0 else 0
med_pct = (medium_risk_count / total_students) * 100 if total_students > 0 else 0
low_pct = (low_risk_count / total_students) * 100 if total_students > 0 else 0

avg_attendance = df_weekly["avg_attendance"].mean() if not df_weekly.empty else 90.5
avg_pred_score = df_cohort["predicted_score"].mean()

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown(
        get_kpi_html("Total Students", f"{total_students}", f"{selected_scenario}"),
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        get_kpi_html("High Risk", f"{high_risk_count}", f"{high_pct:.1f}% of cohort", border_color="#ef4444"),
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        get_kpi_html("Medium Risk", f"{medium_risk_count}", f"{med_pct:.1f}% of cohort", border_color="#f59e0b"),
        unsafe_allow_html=True,
    )
with col4:
    st.markdown(
        get_kpi_html("Low Risk", f"{low_risk_count}", f"{low_pct:.1f}% of cohort", border_color="#10b981"),
        unsafe_allow_html=True,
    )
with col5:
    st.markdown(
        get_kpi_html("Avg Final Score", f"{avg_pred_score:.2f} / 20", f"Pass Mark: {RISK_PASS_MARK}"),
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)


# ==============================================================================
# 4. Interactive Analytical Visualizations (Plotly)
# ==============================================================================
st.markdown("<div class='section-header'>📊 Cohort Risk & Performance Analytics</div>", unsafe_allow_html=True)

row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    # 1. Risk Tier Distribution Donut Chart
    risk_counts = df_cohort["risk_category"].value_counts().reset_index()
    risk_counts.columns = ["risk_category", "count"]
    color_map = {"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"}

    fig_pie = px.pie(
        risk_counts,
        values="count",
        names="risk_category",
        hole=0.55,
        color="risk_category",
        color_discrete_map=color_map,
    )
    fig_pie.update_traces(
        textposition="inside",
        textinfo="percent+label",
        marker=dict(line=dict(color="#ffffff", width=2)),
    )
    apply_plotly_theme(fig_pie, f"Risk Category Distribution ({selected_scenario})")
    st.plotly_chart(fig_pie, use_container_width=True)

with row1_col2:
    # 2. Grade Distribution (Actual G3 vs Predicted Score)
    fig_grade = go.Figure()
    fig_grade.add_trace(
        go.Histogram(
            x=df_cohort["G3"],
            name="Actual G3 Exam Marks",
            opacity=0.65,
            marker_color="#2563eb",
            nbinsx=20,
        )
    )
    fig_grade.add_trace(
        go.Histogram(
            x=df_cohort["predicted_score"],
            name="Model Predicted Score",
            opacity=0.65,
            marker_color="#0d9488",
            nbinsx=20,
        )
    )
    fig_grade.add_vline(
        x=RISK_PASS_MARK,
        line_dash="dash",
        line_color="#ef4444",
        annotation_text="Pass Cutoff (10.0)",
        annotation_position="top right",
    )
    fig_grade.update_layout(barmode="overlay")
    apply_plotly_theme(fig_grade, "Actual vs. Model Predicted Grade Distribution")
    fig_grade.update_xaxes(title_text="Grade Points (0 - 20 Scale)")
    fig_grade.update_yaxes(title_text="Student Count")
    st.plotly_chart(fig_grade, use_container_width=True)


row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    # 3. Absences vs Final Grade Scatter
    fig_abs = px.scatter(
        df_cohort,
        x="absences",
        y="G3",
        color="risk_category",
        color_discrete_map=color_map,
        size="failures",
        size_max=12,
        hover_data=["student_id", "subject", "at_risk_probability"],
    )
    apply_plotly_theme(fig_abs, "Absences vs. Final Grade (Size = Prior Failures)")
    fig_abs.add_hline(y=RISK_PASS_MARK, line_dash="dot", line_color="#ef4444")
    fig_abs.update_xaxes(title_text="Total Semester Absences")
    fig_abs.update_yaxes(title_text="Final Score (G3)")
    st.plotly_chart(fig_abs, use_container_width=True)

with row2_col2:
    # 4. Subject-wise Comparison
    subj_agg = (
        df_cohort.groupby("subject")
        .agg(
            avg_actual=("G3", "mean"),
            avg_pred=("predicted_score", "mean"),
            high_risk_rate=("is_at_risk", "mean"),
            total=("student_id", "count"),
        )
        .reset_index()
    )
    subj_agg["high_risk_pct"] = subj_agg["high_risk_rate"] * 100

    fig_subj = go.Figure(
        data=[
            go.Bar(name="Avg Actual G3", x=subj_agg["subject"], y=subj_agg["avg_actual"], marker_color="#3b82f6"),
            go.Bar(name="Avg Predicted G3", x=subj_agg["subject"], y=subj_agg["avg_pred"], marker_color="#0d9488"),
        ]
    )
    fig_subj.update_layout(barmode="group")
    apply_plotly_theme(fig_subj, "Subject Performance Comparison (Math vs. Portuguese)")
    fig_subj.update_yaxes(title_text="Average Score / 20")
    st.plotly_chart(fig_subj, use_container_width=True)


# ==============================================================================
# 5. Longitudinal Weekly Cohort Trend [SIMULATED]
# ==============================================================================
st.markdown(
    f"<div class='section-header'>📈 Longitudinal Cohort Trajectory (Weeks 1–12) {get_simulated_tag_html()}</div>",
    unsafe_allow_html=True,
)

if not df_weekly.empty:
    fig_week = go.Figure()
    fig_week.add_trace(
        go.Scatter(
            x=df_weekly["week_number"],
            y=df_weekly["avg_attendance"],
            name="Weekly Attendance Rate (%)",
            line=dict(color="#0d9488", width=3),
            mode="lines+markers",
        )
    )
    fig_week.add_trace(
        go.Scatter(
            x=df_weekly["week_number"],
            y=df_weekly["avg_assignment"],
            name="Assignment Completion Rate (%)",
            line=dict(color="#6366f1", width=2, dash="dash"),
            mode="lines+markers",
        )
    )
    fig_week.add_trace(
        go.Scatter(
            x=df_weekly["week_number"],
            y=df_weekly["avg_quiz"] * 5.0,  # Scale 0-20 to 0-100% for shared axis
            name="Quiz Score Trend (Scaled %)",
            line=dict(color="#f59e0b", width=2),
            mode="lines",
        )
    )
    apply_plotly_theme(fig_week, "Longitudinal Cohort Averages Across Semester [SIMULATED DATA]")
    fig_week.update_xaxes(title_text="Academic Week", dtick=1)
    fig_week.update_yaxes(title_text="Percentage Score / Completion (%)")
    st.plotly_chart(fig_week, use_container_width=True)


# ==============================================================================
# 6. High-Risk Student Triage Table
# ==============================================================================
st.markdown("<div class='section-header'>🚨 High-Risk Priority Triage Table</div>", unsafe_allow_html=True)

filter_col1, filter_col2 = st.columns([1, 3])
with filter_col1:
    risk_filter = st.selectbox("Filter Risk Category", ["High", "Medium", "All"], index=0)

filtered_cohort = df_cohort if risk_filter == "All" else df_cohort[df_cohort["risk_category"] == risk_filter]

triage_cols = [
    "student_id", "school", "subject", "failures", "absences", 
    "at_risk_probability", "predicted_score", "G3", "risk_category"
]
display_triage = filtered_cohort[triage_cols].sort_values(by="at_risk_probability", ascending=False).copy()
display_triage.columns = [
    "Student ID", "School", "Subject", "Failures", "Absences", 
    "At-Risk Prob", "Pred Score", "Actual G3", "Risk Tier"
]
display_triage["At-Risk Prob"] = display_triage["At-Risk Prob"].apply(lambda p: f"{p * 100:.1f}%")
display_triage["Pred Score"] = display_triage["Pred Score"].round(2)

st.dataframe(display_triage, use_container_width=True, hide_index=True)


# ==============================================================================
# 7. Recent Early-Warning Alerts Feed
# ==============================================================================
st.markdown("<div class='section-header'>🔔 Recent Early-Warning Monitoring Alerts</div>", unsafe_allow_html=True)

if not df_alerts.empty:
    for _, alert_row in df_alerts.head(6).iterrows():
        alert_id = int(alert_row["id"])
        st_id = alert_row["student_id"]
        a_type = alert_row["alert_type"]
        reason = alert_row["reason"]
        risk_lvl = alert_row["risk_level"]
        status = alert_row["status"]

        badge = get_risk_badge_html(risk_lvl)
        status_color = "#10b981" if status == "Reviewed" else "#ef4444"

        col_a1, col_a2 = st.columns([5, 1])
        with col_a1:
            st.markdown(
                f"""
                <div style='background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px 16px; margin-bottom: 8px;'>
                    <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;'>
                        <span style='font-weight: 700; color: #0f172a;'>Student {st_id} — {a_type}</span>
                        <div>{badge} <span style='font-size: 0.75rem; color: {status_color}; font-weight: 600;'>[{status}]</span></div>
                    </div>
                    <div style='font-size: 0.85rem; color: #475569;'>{reason}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_a2:
            if status == "Open":
                if st.button("Acknowledge", key=f"ack_{alert_id}"):
                    update_alert_status(alert_id, "Reviewed")
                    st.cache_data.clear()
                    st.rerun()
else:
    st.info("No active alerts currently recorded.")


# ==============================================================================
# 8. Data Management Section: Bulk CSV Upload & Student Management
# ==============================================================================
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("<div class='section-header'>📁 Data Management & Student Registry</div>", unsafe_allow_html=True)

tab_upload, tab_add, tab_update = st.tabs(["📤 Bulk CSV Upload", "➕ Add Student", "✏️ Update Student Record"])

with tab_upload:
    st.markdown("Upload student CSV records for automated schema validation, range checks, and database ingestion.")
    uploaded_file = st.file_uploader("Select CSV File", type=["csv"])

    if uploaded_file is not None:
        # Save temp file
        temp_csv_path = Path("temp_upload.csv")
        with open(temp_csv_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if st.button("Execute Bulk Import & Validation"):
            with st.spinner("Validating constraints, duplicate IDs, and marks ranges..."):
                report = bulk_import_csv(temp_csv_path)

            temp_csv_path.unlink(missing_ok=True)

            if report["status"] == "success":
                st.success(f"✅ {report['summary']}")
                st.metric("Successfully Imported", report["imported_count"])
                st.cache_data.clear()
            else:
                st.error(f"❌ {report['summary']}")

            if report["errors"]:
                with st.expander("View Detailed Validation Errors / Warnings"):
                    for err in report["errors"][:20]:
                        st.caption(f"• {err}")

with tab_add:
    st.markdown("Register a new student record with demographic and baseline profile attributes.")
    with st.form("add_student_form"):
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            new_id = st.text_input("Student ID*", placeholder="EDU-0999")
            new_school = st.selectbox("School", ["GP", "MS"])
            new_sex = st.selectbox("Sex", ["F", "M"])
            new_age = st.number_input("Age (10 - 30)*", min_value=10, max_value=30, value=16)
        with f_col2:
            new_address = st.selectbox("Address", ["U", "R"])
            new_famsize = st.selectbox("Family Size", ["GT3", "LE3"])
            new_pstatus = st.selectbox("Parent Status", ["T", "A"])
            new_medu = st.number_input("Mother Education (0-4)", min_value=0, max_value=4, value=2)
        with f_col3:
            new_fedu = st.number_input("Father Education (0-4)", min_value=0, max_value=4, value=2)
            new_mjob = st.selectbox("Mother Job", ["other", "services", "at_home", "teacher", "health"])
            new_fjob = st.selectbox("Father Job", ["other", "services", "at_home", "teacher", "health"])
            new_reason = st.selectbox("Reason", ["course", "home", "reputation", "other"])

        submitted_add = st.form_submit_button("Register Student")
        if submitted_add:
            if not new_id.strip():
                st.error("Student ID is mandatory.")
            else:
                try:
                    add_student({
                        "student_id": new_id.strip(),
                        "school": new_school,
                        "sex": new_sex,
                        "age": int(new_age),
                        "address": new_address,
                        "famsize": new_famsize,
                        "Pstatus": new_pstatus,
                        "Medu": int(new_medu),
                        "Fedu": int(new_fedu),
                        "Mjob": new_mjob,
                        "Fjob": new_fjob,
                        "reason": new_reason,
                        "data_source": "REAL_UCI",
                    })
                    st.success(f"Student '{new_id}' registered successfully.")
                    st.cache_data.clear()
                except Exception as e:
                    st.error(f"Registration failed: {str(e)}")

with tab_update:
    st.markdown("Update mutable attributes for an existing student record.")
    with st.form("update_student_form"):
        u_col1, u_col2 = st.columns(2)
        with u_col1:
            u_id = st.text_input("Existing Student ID*", placeholder="EDU-0001")
            u_age = st.number_input("Updated Age", min_value=10, max_value=30, value=17)
        with u_col2:
            u_address = st.selectbox("Updated Address", ["U", "R"])
            u_reason = st.selectbox("Updated Reason", ["course", "home", "reputation", "other"])

        submitted_update = st.form_submit_button("Update Student Record")
        if submitted_update:
            if not u_id.strip():
                st.error("Please enter a valid Student ID.")
            else:
                res = update_student(u_id.strip(), {
                    "age": int(u_age),
                    "address": u_address,
                    "reason": u_reason,
                })
                if res:
                    st.success(f"Student '{u_id}' updated successfully.")
                    st.cache_data.clear()
                else:
                    st.error(f"Student ID '{u_id}' not found.")

