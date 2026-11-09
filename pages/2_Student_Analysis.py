"""
EduGuard AI — Student Diagnostic Analysis & SHAP Explainability (Stage 7)

Provides individual student-level deep dives:
1. Student Profile & Demographic attributes.
2. Calibrated At-Risk Probability & Predicted Grade (Scenario A and Scenario B).
3. Real TreeSHAP Waterfall plot explaining exact positive risk drivers & protective factors.
4. Factual plain-English advisory diagnostic summary.
5. Longitudinal weekly history chart (Plotly) explicitly labeled SIMULATED.
6. Active alerts queue and counselor intervention logging.
"""

from pathlib import Path
import sys

# Ensure project root is on sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import select
import streamlit as st

from src.config import DATABASE_PATH, RISK_PASS_MARK
from src.database import (
    AcademicRecord,
    Alert,
    Intervention,
    Prediction,
    Student,
    WeeklyPerformance,
    get_db_session,
    update_alert_status,
)
from src.explainability import (
    compute_shap_explanation,
    explain_student,
    get_cached_explainer,
    get_transformed_features,
    waterfall,
)
from src.recommendations import get_recommendations
from src.utils import (
    apply_plotly_theme,
    get_advisory_notice_html,
    get_risk_badge_html,
    get_simulated_tag_html,
    load_custom_css,
)

# Page configuration
st.set_page_config(
    page_title="EduGuard AI | Student Analysis",
    page_icon="👤",
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
            👤 Individual Student Analysis & Diagnostic Triage
        </h2>
        <p style='color: #64748b; font-size: 0.95rem; margin: 0;'>
            Detailed risk diagnosis, game-theoretic SHAP feature attribution, and longitudinal tracking.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(get_advisory_notice_html(), unsafe_allow_html=True)


# ==============================================================================
# 2. Cached Database & Model Retrieval
# ==============================================================================
@st.cache_data(ttl=60)
def get_all_student_ids() -> list:
    """Fetch all unique student IDs from the database."""
    if not DATABASE_PATH.exists():
        return []
    with get_db_session() as session:
        ids = list(session.scalars(select(Student.student_id).order_by(Student.student_id)).all())
    return ids


@st.cache_data(ttl=30)
def load_student_full_profile(student_id: str):
    """Retrieve Student, AcademicRecord, and Predictions for a single student."""
    if not DATABASE_PATH.exists():
        return None

    with get_db_session() as session:
        st_obj = session.scalar(select(Student).where(Student.student_id == student_id))
        if not st_obj:
            return None

        acad_obj = session.scalar(select(AcademicRecord).where(AcademicRecord.student_id == student_id))
        preds = list(session.scalars(select(Prediction).where(Prediction.student_id == student_id)).all())
        weekly = list(
            session.scalars(
                select(WeeklyPerformance)
                .where(WeeklyPerformance.student_id == student_id)
                .order_by(WeeklyPerformance.week_number)
            ).all()
        )
        alerts = list(
            session.scalars(
                select(Alert)
                .where(Alert.student_id == student_id)
                .order_by(Alert.id.desc())
            ).all()
        )

        return {
            "student": {
                "student_id": st_obj.student_id,
                "school": st_obj.school,
                "sex": st_obj.sex,
                "age": st_obj.age,
                "address": st_obj.address,
                "famsize": st_obj.famsize,
                "Pstatus": st_obj.Pstatus,
                "Medu": st_obj.Medu,
                "Fedu": st_obj.Fedu,
                "Mjob": st_obj.Mjob,
                "Fjob": st_obj.Fjob,
                "reason": st_obj.reason,
                "guardian": st_obj.guardian,
                "student_key": st_obj.student_key,
                "data_source": st_obj.data_source,
            },
            "academic": {
                "subject": acad_obj.subject if acad_obj else "math",
                "studytime": acad_obj.studytime if acad_obj else 2,
                "failures": acad_obj.failures if acad_obj else 0,
                "absences": acad_obj.absences if acad_obj else 0,
                "traveltime": acad_obj.traveltime if acad_obj else 1,
                "famrel": acad_obj.famrel if acad_obj else 4,
                "freetime": acad_obj.freetime if acad_obj else 3,
                "goout": acad_obj.goout if acad_obj else 3,
                "Dalc": acad_obj.Dalc if acad_obj else 1,
                "Walc": acad_obj.Walc if acad_obj else 1,
                "health": acad_obj.health if acad_obj else 3,
                "G1": acad_obj.G1 if acad_obj else None,
                "G2": acad_obj.G2 if acad_obj else None,
                "G3": acad_obj.G3 if acad_obj else None,
                "schoolsup": acad_obj.schoolsup if acad_obj else "no",
                "famsup": acad_obj.famsup if acad_obj else "no",
                "paid": acad_obj.paid if acad_obj else "no",
                "activities": acad_obj.activities if acad_obj else "no",
                "nursery": acad_obj.nursery if acad_obj else "yes",
                "higher": acad_obj.higher if acad_obj else "yes",
                "internet": acad_obj.internet if acad_obj else "yes",
                "romantic": acad_obj.romantic if acad_obj else "no",
            } if acad_obj else {},
            "predictions": {p.scenario: p for p in preds},
            "weekly": [
                {
                    "week": w.week_number,
                    "attendance": w.attendance_rate,
                    "quiz": w.quiz_average,
                    "assignment": w.assignment_completion,
                    "hours": w.study_hours,
                    "trajectory": w.trajectory_type,
                    "notes": w.notes,
                }
                for w in weekly
            ],
            "alerts": [
                {
                    "id": a.id,
                    "type": a.alert_type,
                    "reason": a.reason,
                    "risk_level": a.risk_level,
                    "status": a.status,
                    "date": a.alert_date.strftime("%Y-%m-%d"),
                }
                for a in alerts
            ],
        }


@st.cache_resource
def load_explainer_resources(scenario_code: str = "B"):
    """Load and cache the TreeExplainer, preprocessor, and feature names."""
    return get_cached_explainer(scenario_code, "classification")


# ==============================================================================
# 3. Student Search Bar & Scenario Toggle
# ==============================================================================
student_id_list = get_all_student_ids()

if not student_id_list:
    st.error("No student records found. Run `python scripts/seed_db.py` to seed records.")
    st.stop()

col_s1, col_s2 = st.columns([3, 1])

with col_s1:
    selected_student_id = st.selectbox(
        "Search or Select Student ID",
        options=student_id_list,
        index=0,
        help="Type or select a student ID (e.g. EDU-0001, EDU-0002).",
    )

with col_s2:
    selected_scenario_label = st.selectbox(
        "Evaluation Scenario",
        options=["Scenario B (Mid-Semester, with G1/G2)", "Scenario A (Early-Warning, no G1/G2)"],
        index=0,
    )
    scenario_key = "Scenario B" if "Scenario B" in selected_scenario_label else "Scenario A"
    scenario_code = "B" if "Scenario B" in selected_scenario_label else "A"


# Fetch student data
profile = load_student_full_profile(selected_student_id)

if not profile:
    st.error(f"Student '{selected_student_id}' not found.")
    st.stop()

st_data = profile["student"]
acad_data = profile["academic"]
pred_obj = profile["predictions"].get(scenario_key)


# ==============================================================================
# 4. Student Demographics & Academic Profile Card
# ==============================================================================
st.markdown("<div class='section-header'>📋 Student Demographic & Academic Profile</div>", unsafe_allow_html=True)

col_p1, col_p2, col_p3 = st.columns(3)

with col_p1:
    st.markdown(
        f"""
        <div class='eduguard-card'>
            <div style='font-weight: 700; color: #0f172a; margin-bottom: 8px;'>Identity & School</div>
            <div><strong>Student ID:</strong> {st_data['student_id']}</div>
            <div><strong>School:</strong> {st_data['school']} ({'Gabriel Pereira' if st_data['school'] == 'GP' else 'Mousinho da Silveira'})</div>
            <div><strong>Subject:</strong> {acad_data.get('subject', 'N/A').title()}</div>
            <div><strong>Age:</strong> {st_data['age']} years old</div>
            <div><strong>Gender:</strong> {'Female' if st_data['sex'] == 'F' else 'Male'}</div>
            <div><strong>Address:</strong> {'Urban' if st_data['address'] == 'U' else 'Rural'}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_p2:
    study_labels = {1: "< 2 hours", 2: "2 - 5 hours", 3: "5 - 10 hours", 4: "> 10 hours"}
    st.markdown(
        f"""
        <div class='eduguard-card'>
            <div style='font-weight: 700; color: #0f172a; margin-bottom: 8px;'>Habits & Attendance</div>
            <div><strong>Weekly Study Time:</strong> {study_labels.get(acad_data.get('studytime', 2), '2-5h')}</div>
            <div><strong>Historical Absences:</strong> {acad_data.get('absences', 0)} days</div>
            <div><strong>Prior Course Failures:</strong> {acad_data.get('failures', 0)}</div>
            <div><strong>Extra School Support:</strong> {acad_data.get('schoolsup', 'no').capitalize()}</div>
            <div><strong>Family Academic Support:</strong> {acad_data.get('famsup', 'no').capitalize()}</div>
            <div><strong>Internet Access at Home:</strong> {acad_data.get('internet', 'yes').capitalize()}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_p3:
    g1_val = f"{acad_data.get('G1', 0):.1f} / 20" if acad_data.get("G1") is not None else "N/A"
    g2_val = f"{acad_data.get('G2', 0):.1f} / 20" if acad_data.get("G2") is not None else "N/A"
    g3_val = f"{acad_data.get('G3', 0):.1f} / 20" if acad_data.get("G3") is not None else "Pending"

    st.markdown(
        f"""
        <div class='eduguard-card'>
            <div style='font-weight: 700; color: #0f172a; margin-bottom: 8px;'>Examination Ledger</div>
            <div><strong>Period 1 Exam (G1):</strong> {g1_val}</div>
            <div><strong>Period 2 Exam (G2):</strong> {g2_val}</div>
            <div><strong>Final Target Exam (G3):</strong> <strong>{g3_val}</strong></div>
            <div><strong>Pass Requirement:</strong> >= {RISK_PASS_MARK} points</div>
            <div style='margin-top: 8px;'><strong>Data Provenance:</strong> <span class='badge-low'>REAL_UCI</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# 5. Risk Assessment & Model Inference Card
# ==============================================================================
st.markdown("<div class='section-header'>🎯 Model Risk Stratification & Score Forecast</div>", unsafe_allow_html=True)

if pred_obj:
    prob = pred_obj.at_risk_probability
    pred_score = pred_obj.predicted_score
    risk_cat = pred_obj.risk_category

    col_r1, col_r2, col_r3 = st.columns(3)

    with col_r1:
        st.markdown(
            f"""
            <div class='kpi-container' style='border-top-color: {"#ef4444" if risk_cat == "High" else "#f59e0b" if risk_cat == "Medium" else "#10b981"};'>
                <div class='kpi-title'>Assessed Risk Tier</div>
                <div style='margin: 8px 0;'>{get_risk_badge_html(risk_cat)}</div>
                <div class='kpi-subtitle'>Evaluation: {scenario_key}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_r2:
        prob_color = "#ef4444" if prob >= 0.6 else "#f59e0b" if prob >= 0.3 else "#10b981"
        st.markdown(
            f"""
            <div class='kpi-container' style='border-top-color: {prob_color};'>
                <div class='kpi-title'>Calibrated At-Risk Probability</div>
                <div class='kpi-value' style='color: {prob_color};'>{prob * 100:.1f}%</div>
                <div class='kpi-subtitle'>Cutoffs: Low &lt; 30% | High &gt; 60%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_r3:
        st.markdown(
            f"""
            <div class='kpi-container' style='border-top-color: #0d9488;'>
                <div class='kpi-title'>Predicted Final Score</div>
                <div class='kpi-value'>{pred_score:.2f} <span style='font-size: 1rem; color: #64748b;'>/ 20</span></div>
                <div class='kpi-subtitle'>Passing mark: 10.00 / 20.00</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    st.warning(f"No prediction record found for {selected_student_id} under {scenario_key}.")


# ==============================================================================
# 6. Real TreeSHAP Explainability & Waterfall Diagnostic
# ==============================================================================
st.markdown("<div class='section-header'>🔍 SHAP Explainability & Feature Attribution</div>", unsafe_allow_html=True)

# Build student feature dict for explainability
student_full_dict = {**st_data, **acad_data}
if "G1" in acad_data and "G2" in acad_data and acad_data["G1"] is not None and acad_data["G2"] is not None:
    student_full_dict["grade_trend"] = acad_data["G2"] - acad_data["G1"]
    student_full_dict["g1_passing"] = 1 if acad_data["G1"] >= 10.0 else 0
    student_full_dict["g2_passing"] = 1 if acad_data["G2"] >= 10.0 else 0

student_full_dict["alcohol_total"] = acad_data.get("Dalc", 1) + acad_data.get("Walc", 1)
student_full_dict["high_absence"] = 1 if acad_data.get("absences", 0) > 10 else 0
student_full_dict["failures_flag"] = 1 if acad_data.get("failures", 0) > 0 else 0
student_full_dict["study_time_ratio"] = acad_data.get("studytime", 2) / (acad_data.get("freetime", 3) + 1e-5)

with st.spinner("Computing real TreeSHAP game-theoretic attributions..."):
    try:
        explainer, preprocessor, feature_names = load_explainer_resources(scenario_code)
        explanation = explain_student(
            row=pd.Series(student_full_dict),
            explainer=explainer,
            preprocessor=preprocessor,
            feature_names=feature_names,
            top_n=4,
            is_classifier=True,
        )
    except Exception as e:
        explanation = None
        st.error(f"SHAP explanation computation error: {str(e)}")

if explanation:
    # Display the Plain-English Summary Box
    st.markdown(
        f"""
        <div style='background-color: #f0fdfa; border-left: 5px solid #0d9488; padding: 14px 18px; border-radius: 0 8px 8px 0; margin-bottom: 16px;'>
            <div style='font-weight: 700; color: #134e4a; font-size: 0.90rem; margin-bottom: 4px;'>
                📢 Automated Counselor Diagnostic Summary (Derived from Real SHAP Values)
            </div>
            <p style='color: #0f766e; font-size: 0.88rem; line-height: 1.5; margin: 0;'>
                {explanation['plain_english_summary']}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_w1, col_w2 = st.columns([1, 1])

    with col_w1:
        st.markdown("<b>Features Elevating Academic Risk:</b>", unsafe_allow_html=True)
        top_pos = explanation["top_positive"]
        if not top_pos.empty:
            df_pos_show = top_pos[["feature", "shap_value"]].copy()
            df_pos_show["shap_value"] = df_pos_show["shap_value"].apply(lambda v: f"+{v * 100:.2f}% risk")
            df_pos_show.columns = ["Feature", "Risk Contribution"]
            st.dataframe(df_pos_show, use_container_width=True, hide_index=True)
        else:
            st.info("No strong features elevating risk.")

    with col_w2:
        st.markdown("<b>Protective Factors Mitigating Risk:</b>", unsafe_allow_html=True)
        top_neg = explanation["top_negative"]
        if not top_neg.empty:
            df_neg_show = top_neg[["feature", "shap_value"]].copy()
            df_neg_show["shap_value"] = df_neg_show["shap_value"].apply(lambda v: f"{v * 100:.2f}% risk")
            df_neg_show.columns = ["Feature", "Protective Contribution"]
            st.dataframe(df_neg_show, use_container_width=True, hide_index=True)
        else:
            st.info("No strong protective factors identified.")

    # Waterfall plot
    with st.expander("📊 View Interactive Matplotlib SHAP Waterfall Plot", expanded=True):
        X_trans, names = get_transformed_features(preprocessor, pd.DataFrame([student_full_dict]), auto_drop_leakage=True)
        exp_obj = compute_shap_explanation(explainer, X_trans, names, is_classifier=True)
        
        fig, ax = plt.subplots(figsize=(9, 5))
        waterfall(explainer, X_trans, row_idx=0, feature_names=names, is_classifier=True, max_display=10, show=False)
        st.pyplot(plt.gcf())
        plt.close("all")


# ==============================================================================
# 7. Interactive Analytical Profile (Digital Twin Representation)
# ==============================================================================
st.markdown("<div class='section-header'>🧬 Interactive Analytical Profile (Digital Twin View)</div>", unsafe_allow_html=True)

st.markdown(
    """
    <div style='background-color: #f8fafc; border-left: 5px solid #6366f1; padding: 14px 18px; border-radius: 0 8px 8px 0; margin-bottom: 20px;'>
        <div style='font-weight: 700; color: #312e81; font-size: 0.90rem; margin-bottom: 4px;'>
            💡 Educational Decision-Support Profile (Not a Deterministic Digital Twin)
        </div>
        <p style='color: #4338ca; font-size: 0.86rem; line-height: 1.5; margin: 0;'>
            This view synthesizes the student's historical academic background, longitudinal attendance curves,
            simulated risk trajectory over time, current machine learning forecast, and recommended interventions.
            In accordance with educational AI ethics, this is an <strong>interactive analytical profile</strong>
            for counselor decision support, not an exhaustive or deterministic biological/behavioral digital twin.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

twin_col1, twin_col2 = st.columns([1, 1], gap="medium")

with twin_col1:
    st.markdown("<b>Historical Attributes & Academic Baseline:</b>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class='eduguard-card'>
            <div><strong>Student ID:</strong> {st_data['student_id']} ({st_data['school']})</div>
            <div><strong>Academic Subject:</strong> {acad_data.get('subject', 'math').title()}</div>
            <div><strong>Demographics:</strong> {st_data['age']} years old • {'Female' if st_data['sex'] == 'F' else 'Male'} • {'Urban' if st_data['address'] == 'U' else 'Rural'}</div>
            <div><strong>Prior Failures:</strong> {acad_data.get('failures', 0)} • <strong>Total Absences:</strong> {acad_data.get('absences', 0)} days</div>
            <div><strong>Study Commitment:</strong> {study_labels.get(acad_data.get('studytime', 2), '2-5h')} weekly</div>
            <div><strong>Institutional Scaffolding:</strong> School support: {acad_data.get('schoolsup', 'no')}, Family support: {acad_data.get('famsup', 'no')}</div>
            <div><strong>Exam History:</strong> G1: {acad_data.get('G1', 0.0):.1f} / 20 | G2: {acad_data.get('G2', 0.0):.1f} / 20 (Trend: {student_full_dict.get('grade_trend', 0.0):+.1f} pts)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with twin_col2:
    st.markdown("<b>Current Diagnostic State & Attainment Forecast:</b>", unsafe_allow_html=True)
    curr_prob = float(pred_obj.at_risk_probability) if pred_obj and pred_obj.at_risk_probability is not None else 0.50
    curr_score = float(pred_obj.predicted_score) if pred_obj and pred_obj.predicted_score is not None else 10.0
    curr_cat = str(pred_obj.risk_category) if pred_obj and pred_obj.risk_category else "Medium"

    st.markdown(
        f"""
        <div class='eduguard-card'>
            <div><strong>Evaluated Under:</strong> {scenario_key}</div>
            <div><strong>Current Predicted Grade (G3):</strong> <span style='font-size: 1.15rem; font-weight: 800; color: #0d9488;'>{curr_score:.2f} / 20</span></div>
            <div><strong>Calibrated At-Risk Probability:</strong> <span style='font-size: 1.05rem; font-weight: 700; color: {'#ef4444' if curr_prob >= 0.6 else '#f59e0b' if curr_prob >= 0.3 else '#10b981'};'>{curr_prob * 100:.1f}%</span></div>
            <div style='margin: 8px 0;'><strong>Operational Tier:</strong> {get_risk_badge_html(curr_cat)}</div>
            <div><strong>Pass Margin:</strong> {'+' if curr_score >= 10.0 else ''}{curr_score - 10.0:.2f} points vs. 10.0 pass line</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Risk & Attendance Trajectory Over Time (Plotly Chart)
weekly_history = profile.get("weekly", [])
if weekly_history:
    df_w = pd.DataFrame(weekly_history)
    # Estimate longitudinal risk over time across weeks
    df_w["est_risk_prob"] = df_w.apply(
        lambda r: np.clip(curr_prob + (85.0 - r["attendance"]) * 0.008 + (10.0 - r["quiz"]) * 0.02, 0.02, 0.98),
        axis=1,
    )

    fig_twin = go.Figure()
    fig_twin.add_trace(
        go.Scatter(
            x=df_w["week"],
            y=df_w["attendance"],
            name="Attendance Rate (%)",
            line=dict(color="#0d9488", width=3),
            mode="lines+markers",
        )
    )
    fig_twin.add_trace(
        go.Scatter(
            x=df_w["week"],
            y=df_w["est_risk_prob"] * 100.0,
            name="Estimated Risk Probability (%)",
            line=dict(color="#ef4444", width=2.5, dash="dash"),
            mode="lines+markers",
        )
    )
    fig_twin.add_trace(
        go.Scatter(
            x=df_w["week"],
            y=df_w["quiz"] * 5.0,
            name="Weekly Quiz Average (Scaled %)",
            line=dict(color="#f59e0b", width=2),
            mode="lines+markers",
        )
    )
    fig_twin.add_hline(y=75.0, line_dash="dot", line_color="#cbd5e1", annotation_text="Attendance Floor (75%)")
    apply_plotly_theme(fig_twin, f"Analytical Profile: Attendance vs. Risk Probability Over Time [SIMULATED]")
    fig_twin.update_xaxes(title_text="Academic Week Number", dtick=1)
    fig_twin.update_yaxes(title_text="Percentage (%)", range=[0, 105])
    st.plotly_chart(fig_twin, use_container_width=True)

# Current Recommendations Preview
st.markdown("<b>Current Active Recommendations for this Profile:</b>", unsafe_allow_html=True)
shap_list = []
if explanation and not explanation["top_positive"].empty:
    for _, r in explanation["top_positive"].iterrows():
        shap_list.append({"feature": str(r["feature"]), "shap_value": float(r["shap_value"]), "direction": "Elevates Risk"})

rec_plan = get_recommendations(
    profile=student_full_dict,
    risk_result={"at_risk_prob": curr_prob, "risk_tier": curr_cat, "predicted_score": curr_score},
    top_shap_factors=shap_list,
    use_gemini=False,  # Fast deterministic preview
)
rec_c1, rec_c2 = st.columns([2, 1])
with rec_c1:
    st.info(f"**Weekly Focus Theme:** {rec_plan.weekly_plan}\n\n**Revision Strategy:** {rec_plan.revision_strategy}")
with rec_c2:
    st.markdown("**Identified Concepts/Gaps:**")
    for g in rec_plan.learning_gaps[:2]:
        st.markdown(f"- 📌 {g}")
    st.caption("Access full 7-day plan & Gemini synthesis in **4_AI_Recommendations**.")


# ==============================================================================
# 8. Longitudinal Weekly History [SIMULATED]
# ==============================================================================
st.markdown(
    f"<div class='section-header'>📈 Full Weekly Observation Logs (Weeks 1–12) {get_simulated_tag_html()}</div>",
    unsafe_allow_html=True,
)

weekly_history = profile.get("weekly", [])
if weekly_history:
    df_w = pd.DataFrame(weekly_history)

    fig_student_w = go.Figure()
    fig_student_w.add_trace(
        go.Scatter(
            x=df_w["week"],
            y=df_w["attendance"],
            name="Attendance Rate (%)",
            line=dict(color="#0d9488", width=3),
            mode="lines+markers",
        )
    )
    fig_student_w.add_trace(
        go.Scatter(
            x=df_w["week"],
            y=df_w["assignment"],
            name="Assignment Completion (%)",
            line=dict(color="#6366f1", width=2, dash="dash"),
            mode="lines+markers",
        )
    )
    fig_student_w.add_trace(
        go.Scatter(
            x=df_w["week"],
            y=df_w["quiz"] * 5.0,  # Scale to percentage
            name="Quiz Average (Scaled %)",
            line=dict(color="#f59e0b", width=2),
            mode="lines+markers",
        )
    )
    fig_student_w.add_trace(
        go.Scatter(
            x=df_w["week"],
            y=df_w["hours"] * 3.0,
            name="Study Hours (Scaled)",
            line=dict(color="#94a3b8", width=1.5, dash="dot"),
            mode="lines",
        )
    )

    traj_label = df_w.iloc[0].get("trajectory", "STABLE")
    apply_plotly_theme(fig_student_w, f"Weekly Trajectory for {selected_student_id} [{traj_label} TRAJECTORY - SIMULATED]")
    fig_student_w.update_xaxes(title_text="Academic Week", dtick=1)
    fig_student_w.update_yaxes(title_text="Score / Percentage (%)")
    st.plotly_chart(fig_student_w, use_container_width=True)
else:
    st.info("No longitudinal weekly records found for this student.")


# ==============================================================================
# 8. Student Alerts & Counselor Intervention Log
# ==============================================================================
st.markdown("<div class='section-header'>🔔 Student Monitoring Alerts & Advisory Interventions</div>", unsafe_allow_html=True)

st_alerts = profile.get("alerts", [])
col_al1, col_al2 = st.columns([1, 1])

with col_al1:
    st.markdown("<b>Active Early-Warning Alerts:</b>", unsafe_allow_html=True)
    if st_alerts:
        for alert in st_alerts:
            a_id = alert["id"]
            a_type = alert["type"]
            a_reason = alert["reason"]
            a_risk = alert["risk_level"]
            a_status = alert["status"]
            a_date = alert["date"]

            badge = get_risk_badge_html(a_risk)
            status_color = "#10b981" if a_status == "Reviewed" else "#ef4444"

            st.markdown(
                f"""
                <div style='background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px 14px; margin-bottom: 8px;'>
                    <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px;'>
                        <span style='font-weight: 700; color: #0f172a; font-size: 0.85rem;'>{a_type} ({a_date})</span>
                        <div>{badge} <span style='font-size: 0.72rem; color: {status_color}; font-weight: 600;'>[{a_status}]</span></div>
                    </div>
                    <div style='font-size: 0.80rem; color: #475569;'>{a_reason}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if a_status == "Open":
                if st.button("Mark as Reviewed", key=f"rev_{a_id}"):
                    update_alert_status(a_id, "Reviewed")
                    st.cache_data.clear()
                    st.rerun()
    else:
        st.success("No active early-warning alerts for this student.")

with col_al2:
    st.markdown("<b>Log Counselor Intervention Action:</b>", unsafe_allow_html=True)
    with st.form("log_intervention_form"):
        int_type = st.selectbox(
            "Intervention Strategy",
            ["MANDATORY_TUTORING", "COUNSELOR_MEETING", "ATTENDANCE_CONTRACT", "PEER_STUDY_GROUP"],
        )
        int_desc = st.text_area("Notes & Action Plan", placeholder="Met with student to discuss exam preparation and attendance...")
        int_assigned = st.text_input("Assigned Staff", value="Academic Advisor")
        submitted_int = st.form_submit_button("Record Intervention")

        if submitted_int:
            if not int_desc.strip():
                st.error("Please enter action notes.")
            else:
                with get_db_session() as session:
                    new_int = Intervention(
                        student_id=selected_student_id,
                        intervention_type=int_type,
                        description=int_desc.strip(),
                        assigned_to=int_assigned.strip(),
                        status="Scheduled",
                        data_source="SIMULATED",
                    )
                    session.add(new_int)
                st.success(f"Intervention '{int_type}' logged successfully.")
                st.cache_data.clear()

