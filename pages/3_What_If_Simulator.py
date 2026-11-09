"""
EduGuard AI — What-If Scenario Simulator (Stage 8)

Interactive decision-support simulator allowing educators and counselors to test
hypothetical adjustments to student attributes using our trained champion models.

CRITICAL ETHICAL & CAUSAL NOTICE:
"Hypothetical model scenario, not a guaranteed outcome or causal effect."
This tool visualizes the model's learned response surface based on historical
correlations. In real-world educational counseling, artificially altering a
factor (e.g., forcing attendance or changing study hours on paper) does not
guarantee an improved grade unless the underlying root causes are addressed.
"""

from pathlib import Path
import sys

# Ensure project root is on sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sqlalchemy import select
import streamlit as st

from src.classification import load_classification_model, predict_risk
from src.config import DATABASE_PATH, RISK_PASS_MARK
from src.database import (
    AcademicRecord,
    Student,
    ensure_database_ready,
    get_db_session,
    get_engine,
)
from src.features import add_engineered_features
from src.regression import load_model, predict_score
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
    page_title="EduGuard AI | What-If Simulator",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Ensure database tables exist and are seeded before any queries run
ensure_database_ready()

load_custom_css()

# ==============================================================================
# 1. Header & Persistent Ethical Governance Banner
# ==============================================================================
st.markdown(
    """
    <div style='margin-bottom: 16px;'>
        <h2 style='color: #0f172a; font-weight: 800; margin-bottom: 4px;'>
            🔮 What-If Scenario Simulator & Sensitivity Triage
        </h2>
        <p style='color: #64748b; font-size: 0.95rem; margin: 0;'>
            Explore how hypothetical changes in academic momentum, attendance, and study habits shift risk estimates.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Mandatory Persistent Causal Notice Banner
st.markdown(
    """
    <div style='background-color: #fffbeb; border-left: 5px solid #f59e0b; padding: 16px 20px; border-radius: 0 10px 10px 0; margin-bottom: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);'>
        <div style='font-weight: 700; color: #92400e; font-size: 0.95rem; margin-bottom: 4px;'>
            ⚠️ Critical Advisory Notice: Hypothetical Model Scenario, Not a Guaranteed Outcome or Causal Effect
        </div>
        <p style='font-size: 0.86rem; color: #78350f; line-height: 1.5; margin: 0;'>
            This simulation demonstrates how the trained machine learning model maps feature changes based on
            <strong>historical statistical associations</strong>. It does <strong>not</strong> prove physical causation.
            Artificially adjusting a student's hours or attendance on paper will not improve final attainment unless
            educators intervene to resolve the student's true underlying barriers (e.g. chronic illness, personal stress,
            or foundational concept gaps).
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# 2. Cached Model & Data Loaders
# ==============================================================================
@st.cache_resource
def load_simulation_models(scenario_code: str = "B"):
    """Load and cache champion classification and regression pipelines."""
    clf, meta_clf = load_classification_model(scenario_code)
    reg, meta_reg = load_model(scenario_code)
    return clf, meta_clf, reg, meta_reg


@st.cache_data(ttl=60)
def get_all_student_profiles() -> pd.DataFrame:
    """Fetch student baseline records from the database for preset loading."""
    engine = get_engine()
    stmt = (
        select(
            Student.student_id,
            Student.school,
            Student.sex,
            Student.age,
            Student.address,
            Student.famsize,
            Student.Pstatus,
            Student.Medu,
            Student.Fedu,
            Student.Mjob,
            Student.Fjob,
            Student.reason,
            Student.guardian,
            AcademicRecord.subject,
            AcademicRecord.traveltime,
            AcademicRecord.studytime,
            AcademicRecord.failures,
            AcademicRecord.schoolsup,
            AcademicRecord.famsup,
            AcademicRecord.paid,
            AcademicRecord.activities,
            AcademicRecord.nursery,
            AcademicRecord.higher,
            AcademicRecord.internet,
            AcademicRecord.romantic,
            AcademicRecord.famrel,
            AcademicRecord.freetime,
            AcademicRecord.goout,
            AcademicRecord.Dalc,
            AcademicRecord.Walc,
            AcademicRecord.health,
            AcademicRecord.absences,
            AcademicRecord.G1,
            AcademicRecord.G2,
            AcademicRecord.G3,
        )
        .join(AcademicRecord, Student.student_id == AcademicRecord.student_id)
        .order_by(Student.student_id)
    )
    try:
        with engine.connect() as conn:
            df = pd.read_sql(stmt, conn)
        return df
    except Exception as exc:
        st.error(f"⚠️ Error loading student profiles from database: {exc}")
        return pd.DataFrame()


df_students = get_all_student_profiles()

if df_students.empty:
    st.error("No student records found in the database. Run `python scripts/seed_db.py` to seed records.")
    st.stop()


# ==============================================================================
# 3. Preset Selection & Scenario Configuration
# ==============================================================================
col_cfg1, col_cfg2 = st.columns([2, 1])

with col_cfg1:
    student_options = ["-- Cohort Median Default --"] + list(df_students["student_id"].unique())
    selected_preset = st.selectbox(
        "Load Student Baseline Profile",
        options=student_options,
        index=1,
        help="Select an existing student to initialize sliders with their real values, or use cohort median defaults.",
    )

with col_cfg2:
    selected_scenario_label = st.selectbox(
        "Modeling Scenario",
        options=["Scenario B (Mid-Semester, with G1/G2)", "Scenario A (Early-Warning, no G1/G2)"],
        index=0,
        help="Scenario B models academic momentum (G1, G2, grade trend). Scenario A uses early habits only.",
    )
    scenario_code = "B" if "Scenario B" in selected_scenario_label else "A"
    is_scenario_b = (scenario_code == "B")

# Load models
with st.spinner(f"Loading {selected_scenario_label} champion models..."):
    clf, meta_clf, reg, meta_reg = load_simulation_models(scenario_code)


# Extract baseline record
if selected_preset == "-- Cohort Median Default --":
    base_record = {
        "school": "GP",
        "subject": "math",
        "traveltime": 1,
        "studytime": 2,
        "failures": 0,
        "schoolsup": "no",
        "famsup": "no",
        "paid": "no",
        "activities": "yes",
        "nursery": "yes",
        "higher": "yes",
        "internet": "yes",
        "romantic": "no",
        "famrel": 4,
        "freetime": 3,
        "goout": 3,
        "Dalc": 1,
        "Walc": 2,
        "health": 3,
        "absences": 4,
        "G1": 11.0,
        "G2": 11.0,
        "G3": 11.0,
    }
    actual_g3_display = "N/A (Synthetic Baseline)"
else:
    match_row = df_students[df_students["student_id"] == selected_preset].iloc[0]
    base_record = match_row.to_dict()
    actual_g3_display = f"{base_record.get('G3', 0.0):.1f} / 20" if pd.notna(base_record.get("G3")) else "Pending"


# ==============================================================================
# 4. Interactive Feature Sliders (Only features the model genuinely contains)
# ==============================================================================
st.markdown("<hr>", unsafe_allow_html=True)
col_input, col_results = st.columns([1, 1], gap="large")

with col_input:
    st.markdown("<div class='section-header'>🎛️ Hypothetical Feature Adjustments</div>", unsafe_allow_html=True)
    st.caption("Adjust sliders below. Sliders are strictly clipped to legitimate operational domain bounds.")

    hypothetical_record = base_record.copy()

    # Section 1: Academic Momentum (Scenario B Only)
    if is_scenario_b:
        with st.expander("📝 1. Preliminary Examination Marks (G1 & G2)", expanded=True):
            init_g1 = float(base_record.get("G1", 11.0)) if pd.notna(base_record.get("G1")) else 11.0
            init_g2 = float(base_record.get("G2", 11.0)) if pd.notna(base_record.get("G2")) else 11.0

            sim_g1 = st.slider(
                "Period 1 Exam (G1) [0.0 - 20.0]",
                min_value=0.0,
                max_value=20.0,
                value=float(np.clip(init_g1, 0.0, 20.0)),
                step=0.5,
                help="First preliminary exam mark.",
            )
            sim_g2 = st.slider(
                "Period 2 Exam (G2) [0.0 - 20.0]",
                min_value=0.0,
                max_value=20.0,
                value=float(np.clip(init_g2, 0.0, 20.0)),
                step=0.5,
                help="Mid-semester preliminary exam mark.",
            )
            hypothetical_record["G1"] = sim_g1
            hypothetical_record["G2"] = sim_g2
            sim_trend = sim_g2 - sim_g1
            st.info(f"Calculated Momentum Trend (G2 - G1): **{sim_trend:+.1f} points**")
    else:
        # In Scenario A, ensure G1/G2 are removed
        hypothetical_record.pop("G1", None)
        hypothetical_record.pop("G2", None)

    # Section 2: Study Habits & Engagement
    with st.expander("📚 2. Attendance & Study Commitment", expanded=True):
        init_abs = int(base_record.get("absences", 4)) if pd.notna(base_record.get("absences")) else 4
        init_study = int(base_record.get("studytime", 2)) if pd.notna(base_record.get("studytime")) else 2
        init_fail = int(base_record.get("failures", 0)) if pd.notna(base_record.get("failures")) else 0

        sim_absences = st.slider(
            "Semester Absences [0 - 50 days]",
            min_value=0,
            max_value=50,
            value=int(np.clip(init_abs, 0, 50)),
            step=1,
            help="Total sessions missed. Absences >= 10 flag chronic absenteeism.",
        )
        sim_studytime = st.select_slider(
            "Weekly Study Time",
            options=[1, 2, 3, 4],
            value=int(np.clip(init_study, 1, 4)),
            format_func=lambda x: {1: "< 2 hours", 2: "2 - 5 hours", 3: "5 - 10 hours", 4: "> 10 hours"}[x],
        )
        sim_failures = st.select_slider(
            "Past Course Failures",
            options=[0, 1, 2, 3],
            value=int(np.clip(init_fail, 0, 3)),
            format_func=lambda x: f"{x} prior failures" if x > 0 else "0 (No prior failures)",
        )

        hypothetical_record["absences"] = sim_absences
        hypothetical_record["studytime"] = sim_studytime
        hypothetical_record["failures"] = sim_failures

    # Section 3: Lifestyle & Social Balance
    with st.expander("🏡 3. Lifestyle & Family Environment", expanded=False):
        init_famrel = int(base_record.get("famrel", 4)) if pd.notna(base_record.get("famrel")) else 4
        init_freetime = int(base_record.get("freetime", 3)) if pd.notna(base_record.get("freetime")) else 3
        init_goout = int(base_record.get("goout", 3)) if pd.notna(base_record.get("goout")) else 3
        init_dalc = int(base_record.get("Dalc", 1)) if pd.notna(base_record.get("Dalc")) else 1
        init_walc = int(base_record.get("Walc", 2)) if pd.notna(base_record.get("Walc")) else 2
        init_health = int(base_record.get("health", 3)) if pd.notna(base_record.get("health")) else 3

        sim_famrel = st.slider("Family Relationship Quality [1 (terrible) - 5 (excellent)]", 1, 5, int(np.clip(init_famrel, 1, 5)))
        sim_freetime = st.slider("Free Time After School [1 (very low) - 5 (very high)]", 1, 5, int(np.clip(init_freetime, 1, 5)))
        sim_goout = st.slider("Going Out with Friends [1 (rarely) - 5 (frequently)]", 1, 5, int(np.clip(init_goout, 1, 5)))
        sim_dalc = st.slider("Workday Alcohol Consumption [1 (very low) - 5 (very high)]", 1, 5, int(np.clip(init_dalc, 1, 5)))
        sim_walc = st.slider("Weekend Alcohol Consumption [1 (very low) - 5 (very high)]", 1, 5, int(np.clip(init_walc, 1, 5)))
        sim_health = st.slider("Current Health Status [1 (very bad) - 5 (very good)]", 1, 5, int(np.clip(init_health, 1, 5)))

        hypothetical_record["famrel"] = sim_famrel
        hypothetical_record["freetime"] = sim_freetime
        hypothetical_record["goout"] = sim_goout
        hypothetical_record["Dalc"] = sim_dalc
        hypothetical_record["Walc"] = sim_walc
        hypothetical_record["health"] = sim_health

    # Section 4: Support & Institutional Environment
    with st.expander("🏫 4. Institutional & Extracurricular Support", expanded=False):
        c_sup1, c_sup2 = st.columns(2)
        with c_sup1:
            sim_schoolsup = st.selectbox("Extra Educational Support", ["no", "yes"], index=1 if base_record.get("schoolsup") == "yes" else 0)
            sim_famsup = st.selectbox("Family Educational Support", ["no", "yes"], index=1 if base_record.get("famsup") == "yes" else 0)
            sim_paid = st.selectbox("Extra Paid Classes", ["no", "yes"], index=1 if base_record.get("paid") == "yes" else 0)
        with c_sup2:
            sim_higher = st.selectbox("Wants Higher Education", ["yes", "no"], index=1 if base_record.get("higher") == "no" else 0)
            sim_internet = st.selectbox("Home Internet Access", ["yes", "no"], index=1 if base_record.get("internet") == "no" else 0)
            sim_subject = st.selectbox("Academic Subject", ["math", "portuguese"], index=1 if base_record.get("subject") == "portuguese" else 0)

        hypothetical_record["schoolsup"] = sim_schoolsup
        hypothetical_record["famsup"] = sim_famsup
        hypothetical_record["paid"] = sim_paid
        hypothetical_record["higher"] = sim_higher
        hypothetical_record["internet"] = sim_internet
        hypothetical_record["subject"] = sim_subject


# ==============================================================================
# 5. Dual Inference & Real-Time Delta Computation
# ==============================================================================
def execute_inference(record_dict: dict, scenario_flag: bool):
    """Run preprocessing, feature engineering, and inference."""
    df_raw = pd.DataFrame([record_dict])

    # Drop administrative and target leakage columns
    drop_candidates = ["G3", "at_risk", "student_key", "student_id", "is_simulated", "data_source"]
    clean_df = df_raw.drop(columns=[c for c in drop_candidates if c in df_raw.columns], errors="ignore")

    # Add engineered features
    engineered_df = add_engineered_features(clean_df, include_trend=scenario_flag)

    # Threshold: 0.15 for B, 0.25 for A
    opt_t = 0.15 if scenario_flag else 0.25

    clf_pred = predict_risk(clf, meta_clf, engineered_df, threshold=opt_t)
    reg_pred = predict_score(reg, meta_reg, engineered_df)

    return {
        "prob": float(clf_pred["probabilities"][0]),
        "score": float(reg_pred[0]),
        "risk_cat": str(clf_pred["risk_tiers"][0]),
        "is_at_risk": int(clf_pred["predictions"][0]),
    }


# Run original vs. hypothetical inference
orig_result = execute_inference(base_record, is_scenario_b)
hypo_result = execute_inference(hypothetical_record, is_scenario_b)

delta_score = hypo_result["score"] - orig_result["score"]
delta_prob = hypo_result["prob"] - orig_result["prob"]


# ==============================================================================
# 6. Before vs. After Results Display & Plotly Visualizations
# ==============================================================================
with col_results:
    st.markdown("<div class='section-header'>📊 Simulated Impact & Comparison</div>", unsafe_allow_html=True)

    # KPI Comparison Summary Cards
    res_kpi1, res_kpi2, res_kpi3 = st.columns(3)

    with res_kpi1:
        st.markdown(
            f"""
            <div class='kpi-container' style='border-top-color: #0d9488;'>
                <div class='kpi-title'>Predicted Final Score</div>
                <div class='kpi-value'>{hypo_result['score']:.2f} <span style='font-size: 0.9rem; color: #64748b;'>/ 20</span></div>
                <div class='kpi-subtitle'>Original: {orig_result['score']:.2f} (Δ {delta_score:+.2f} pts)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with res_kpi2:
        prob_border = "#ef4444" if hypo_result["prob"] >= 0.6 else "#f59e0b" if hypo_result["prob"] >= 0.3 else "#10b981"
        st.markdown(
            f"""
            <div class='kpi-container' style='border-top-color: {prob_border};'>
                <div class='kpi-title'>At-Risk Probability</div>
                <div class='kpi-value' style='color: {prob_border};'>{hypo_result['prob'] * 100:.1f}%</div>
                <div class='kpi-subtitle'>Original: {orig_result['prob'] * 100:.1f}% (Δ {delta_prob * 100:+.1f}%)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with res_kpi3:
        st.markdown(
            f"""
            <div class='kpi-container' style='border-top-color: #6366f1;'>
                <div class='kpi-title'>Hypothetical Risk Tier</div>
                <div style='margin: 8px 0;'>{get_risk_badge_html(hypo_result['risk_cat'])}</div>
                <div class='kpi-subtitle'>Baseline: {orig_result['risk_cat']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Side-by-Side Plotly Comparison Chart
    fig_comp = go.Figure()

    # Score Bars
    fig_comp.add_trace(
        go.Bar(
            name="Predicted Score (0-20)",
            x=["Original Baseline", "Hypothetical Scenario"],
            y=[orig_result["score"], hypo_result["score"]],
            marker_color=["#94a3b8", "#0d9488"],
            text=[f"{orig_result['score']:.2f} pts", f"{hypo_result['score']:.2f} pts"],
            textposition="auto",
        )
    )

    # Probability Bars (scaled to 20 for dual comparison or percentage)
    fig_comp.add_trace(
        go.Bar(
            name="At-Risk Probability (Scaled 0-20)",
            x=["Original Baseline", "Hypothetical Scenario"],
            y=[orig_result["prob"] * 20.0, hypo_result["prob"] * 20.0],
            marker_color=["#cbd5e1", "#f59e0b" if hypo_result["prob"] >= 0.3 else "#10b981"],
            text=[f"{orig_result['prob'] * 100:.1f}%", f"{hypo_result['prob'] * 100:.1f}%"],
            textposition="auto",
        )
    )

    fig_comp.add_hline(y=10.0, line_dash="dash", line_color="#ef4444", annotation_text="Pass Mark (10.0)")
    fig_comp.update_layout(barmode="group", yaxis_range=[0, 21])
    apply_plotly_theme(fig_comp, "Direct Before vs. After Simulation Comparison")
    fig_comp.update_yaxes(title_text="Grade Points / Scaled Probability")
    st.plotly_chart(fig_comp, use_container_width=True)

    # Detailed Delta Ledger
    st.markdown("<b>Detailed Metric Shift Ledger:</b>", unsafe_allow_html=True)
    ledger_data = [
        {
            "Metric": "Final Grade Prediction (G3)",
            "Original Value": f"{orig_result['score']:.2f} / 20",
            "Hypothetical Value": f"{hypo_result['score']:.2f} / 20",
            "Net Shift (Δ)": f"{delta_score:+.2f} points",
            "Evaluation": "Favorable (+)" if delta_score >= 0 else "Unfavorable (-)",
        },
        {
            "Metric": "Calibrated At-Risk Probability",
            "Original Value": f"{orig_result['prob'] * 100:.1f}%",
            "Hypothetical Value": f"{hypo_result['prob'] * 100:.1f}%",
            "Net Shift (Δ)": f"{delta_prob * 100:+.1f}%",
            "Evaluation": "Risk Decreased (Protective)" if delta_prob <= 0 else "Risk Increased (Elevated)",
        },
        {
            "Metric": "Operational Risk Category",
            "Original Value": orig_result["risk_cat"],
            "Hypothetical Value": hypo_result["risk_cat"],
            "Net Shift (Δ)": "Unchanged" if orig_result["risk_cat"] == hypo_result["risk_cat"] else f"{orig_result['risk_cat']} -> {hypo_result['risk_cat']}",
            "Evaluation": "Optimal" if hypo_result["risk_cat"] == "Low" else "Needs Attention",
        },
    ]
    st.dataframe(pd.DataFrame(ledger_data), use_container_width=True, hide_index=True)

    # Narrative Diagnostic Insight
    st.markdown(
        f"""
        <div style='background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px 18px; margin-top: 12px;'>
            <div style='font-weight: 700; color: #0f172a; font-size: 0.88rem; margin-bottom: 4px;'>
                💡 Decision-Support Sensitivity Interpretation
            </div>
            <p style='font-size: 0.84rem; color: #475569; line-height: 1.5; margin: 0;'>
                Under <strong>{selected_scenario_label}</strong>, the hypothetical adjustments produced a net
                <strong>{delta_score:+.2f} grade points</strong> shift in predicted final attainment and a
                <strong>{delta_prob * 100:+.1f}%</strong> change in at-risk probability.
                This indicates that the model is sensitive to these feature adjustments. Counselors should consider
                whether the underlying student challenges (e.g. attendance barriers, test anxiety) can be addressed
                through targeted institutional interventions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

