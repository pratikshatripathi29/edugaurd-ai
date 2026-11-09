"""
EduGuard AI — Main Application Entrypoint (Stage 7)

Interactive Streamlit landing page with ethical governance notices, system
architecture summary, and real-time database integrity metrics.
"""

from pathlib import Path
import sys

# Ensure project root is on sys.path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
from sqlalchemy import func, select
import streamlit as st

from src.config import DATABASE_PATH, RISK_PASS_MARK, RISK_THRESHOLDS
from src.database import Alert, Prediction, Student, WeeklyPerformance, get_db_session
from src.utils import (
    get_advisory_notice_html,
    get_kpi_html,
    get_simulated_data_notice_html,
    load_custom_css,
)

# Page configuration
st.set_page_config(
    page_title="EduGuard AI | Early-Warning & Decision Support",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom navy/white/teal CSS
load_custom_css()

# ==============================================================================
# 1. Sidebar Brand & Navigation Notice
# ==============================================================================
with st.sidebar:
    st.markdown("## 🎓 EduGuard AI")
    st.markdown(
        "<div style='font-size: 0.85rem; color: #94a3b8; margin-bottom: 12px;'>"
        "Responsible Academic Early-Warning System"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("### 📌 Navigation")
    st.info(
        "Use the sidebar navigation above to access:\n\n"
        "• **1_Admin_Dashboard:** Cohort KPIs, grade distributions, and triage alerts.\n\n"
        "• **2_Student_Analysis:** Individual student lookups, SHAP waterfall diagnosis, and analytical profile.\n\n"
        "• **3_What_If_Simulator:** Interactive sensitivity analysis and hypothetical score/risk adjustments.\n\n"
        "• **4_AI_Recommendations:** Supportive 7-day recovery plans powered by Gemini with rule fallback.\n\n"
        "• **5_Early_Warning:** Longitudinal alert triage, rising risk detection, and intervention tracker.\n\n"
        "• **6_Model_Explainability:** Empirical model audits (regression & classification), calibration, SHAP, and fairness."
    )
    st.markdown("---")
    st.markdown("### ⚙️ System Settings")
    st.caption(f"**Database:** SQLite ({DATABASE_PATH.name})")
    st.caption(f"**Passing Grade:** {RISK_PASS_MARK} / 20.0")
    st.caption(f"**Low Risk Cutoff:** < {RISK_THRESHOLDS['low'] * 100:.0f}%")
    st.caption(f"**High Risk Cutoff:** > {RISK_THRESHOLDS['high'] * 100:.0f}%")


# ==============================================================================
# 2. Main Hero & Mission Statement
# ==============================================================================
st.markdown(
    """
    <div style='margin-bottom: 24px;'>
        <h1 style='color: #0f172a; font-weight: 800; font-size: 2.3rem; margin-bottom: 6px;'>
            🎓 EduGuard AI
        </h1>
        <p style='color: #475569; font-size: 1.1rem; margin: 0;'>
            A responsible machine learning decision-support platform designed to detect students
            at academic risk early, explain risk drivers with SHAP, and guide personalized counseling.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Ethical Decision Support Notice
st.markdown(get_advisory_notice_html(), unsafe_allow_html=True)

# Simulated Data Governance Notice
st.markdown(get_simulated_data_notice_html(), unsafe_allow_html=True)


# ==============================================================================
# 3. Live System Telemetry & Database KPIs (Cached)
# ==============================================================================
@st.cache_data(ttl=60)
def load_system_telemetry():
    """Retrieve real-time database counts."""
    if not DATABASE_PATH.exists():
        return None

    try:
        with get_db_session() as session:
            n_students = session.scalar(select(func.count(Student.id))) or 0
            n_weekly = session.scalar(select(func.count(WeeklyPerformance.id))) or 0
            n_preds = session.scalar(select(func.count(Prediction.id))) or 0
            n_alerts = session.scalar(select(func.count(Alert.id))) or 0
            n_open_alerts = (
                session.scalar(select(func.count(Alert.id)).where(Alert.status == "Open"))
                or 0
            )

            # High risk count in Scenario B
            n_high_risk = (
                session.scalar(
                    select(func.count(Prediction.id)).where(
                        Prediction.scenario == "Scenario B",
                        Prediction.risk_category == "High",
                    )
                )
                or 0
            )

        return {
            "students": n_students,
            "weekly_logs": n_weekly,
            "predictions": n_preds,
            "alerts": n_alerts,
            "open_alerts": n_open_alerts,
            "high_risk": n_high_risk,
        }
    except Exception as e:
        return {"error": str(e)}


telemetry = load_system_telemetry()

st.markdown("<div class='section-header'>📊 Active System Telemetry</div>", unsafe_allow_html=True)

if telemetry and "error" not in telemetry:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            get_kpi_html("Registered Students", f"{telemetry['students']:,}", "UCI Held-Out Test Cohort"),
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            get_kpi_html("Active Predictions", f"{telemetry['predictions']:,}", "Scenario A & B Audited"),
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            get_kpi_html(
                "Weekly Logs [SIMULATED]",
                f"{telemetry['weekly_logs']:,}",
                "12 Weeks per Student",
                border_color="#8b5cf6",
            ),
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            get_kpi_html(
                "Active Open Alerts",
                f"{telemetry['open_alerts']:,}",
                f"{telemetry['high_risk']} High Priority Cases",
                border_color="#ef4444",
            ),
            unsafe_allow_html=True,
        )
else:
    st.warning("Database not yet seeded. Run `python scripts/seed_db.py` to populate records.")


# ==============================================================================
# 4. Multi-Page Module Guide Cards
# ==============================================================================
st.markdown("<div class='section-header'>🧭 Platform Capabilities</div>", unsafe_allow_html=True)

col_mod1, col_mod2 = st.columns(2)

with col_mod1:
    st.markdown(
        """
        <div class='eduguard-card'>
            <div style='font-size: 1.15rem; font-weight: 700; color: #0f172a; margin-bottom: 8px;'>
                📈 1. Admin & Cohort Dashboard
            </div>
            <p style='color: #475569; font-size: 0.90rem; line-height: 1.5;'>
                Comprehensive institutional overview for academic department heads, school principals,
                and guidance counselors:
            </p>
            <ul style='color: #334155; font-size: 0.88rem; padding-left: 20px; line-height: 1.6;'>
                <li><strong>Cohort Risk Triage:</strong> Breakdown across Low, Medium, and High risk tiers.</li>
                <li><strong>Grade Distributions:</strong> Actual examination marks vs. regression forecasts.</li>
                <li><strong>Absences & Subject Analysis:</strong> Mathematics vs. Portuguese risk correlations.</li>
                <li><strong>Longitudinal Alert Queue:</strong> Real-time monitoring feed flagging sudden attendance drops and score collapses.</li>
                <li><strong>Data Management:</strong> Bulk CSV import with strict validation and student registration forms.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_mod2:
    st.markdown(
        """
        <div class='eduguard-card'>
            <div style='font-size: 1.15rem; font-weight: 700; color: #0f172a; margin-bottom: 8px;'>
                👤 2. Student Analysis & SHAP Diagnosis
            </div>
            <p style='color: #475569; font-size: 0.90rem; line-height: 1.5;'>
                Individual student diagnostic deep-dive tailored for one-on-one mentor check-ins
                and parent-teacher conferences:
            </p>
            <ul style='color: #334155; font-size: 0.88rem; padding-left: 20px; line-height: 1.6;'>
                <li><strong>Instant Profile Lookup:</strong> Search by Student ID with complete demographic background.</li>
                <li><strong>Scenario Comparison:</strong> Toggle between Scenario A (Early-Warning) and Scenario B (Mid-Semester).</li>
                <li><strong>SHAP Waterfall Attribution:</strong> Interactive game-theoretic visualization of exact positive risk drivers and protective barriers.</li>
                <li><strong>Automated Plain-English Diagnostics:</strong> Multi-sentence factual summary explaining *why* the student was flagged.</li>
                <li><strong>Weekly Trajectory Tracking [SIMULATED]:</strong> Interactive Plotly multi-line history over Weeks 1–12.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

col_mod3, col_mod4 = st.columns(2)

with col_mod3:
    st.markdown(
        """
        <div class='eduguard-card'>
            <div style='font-size: 1.15rem; font-weight: 700; color: #0f172a; margin-bottom: 8px;'>
                🔮 3. What-If Scenario Simulator
            </div>
            <p style='color: #475569; font-size: 0.90rem; line-height: 1.5;'>
                Explore hypothetical shifts in student study habits, attendance, and exam marks:
            </p>
            <ul style='color: #334155; font-size: 0.88rem; padding-left: 20px; line-height: 1.6;'>
                <li><strong>Dual Real-Time Inference:</strong> Compares baseline vs. modified student simultaneously.</li>
                <li><strong>Genuinely Used Features Only:</strong> Sliders restricted to authentic model parameters.</li>
                <li><strong>Interactive Delta Ledger:</strong> Net shift calculations for predicted score and risk probability.</li>
                <li><strong>Ethical Boundary Banner:</strong> Explicit advisory disclaimers on non-causality.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_mod4:
    st.markdown(
        """
        <div class='eduguard-card'>
            <div style='font-size: 1.15rem; font-weight: 700; color: #0f172a; margin-bottom: 8px;'>
                💡 4. AI Recommendations & Intervention Studio
            </div>
            <p style='color: #475569; font-size: 0.90rem; line-height: 1.5;'>
                Actionable 7-day pedagogical recovery plans synthesizing SHAP attributions:
            </p>
            <ul style='color: #334155; font-size: 0.88rem; padding-left: 20px; line-height: 1.6;'>
                <li><strong>Dual Engine Architecture:</strong> Google Gemini 2.5 generative synthesis with deterministic rule-based fallback.</li>
                <li><strong>Strict Privacy Protocol:</strong> Zero PII transmitted (anonymized educational metrics only).</li>
                <li><strong>Comprehensive Structure:</strong> 7-day daily goals, revision strategy, practice exercises, and campus support.</li>
                <li><strong>Human-in-the-Loop Logging:</strong> Editable teacher workspace saving directly to the database Interventions record.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

col_mod5, col_mod6 = st.columns(2)

with col_mod5:
    st.markdown(
        """
        <div class='eduguard-card'>
            <div style='font-size: 1.15rem; font-weight: 700; color: #0f172a; margin-bottom: 8px;'>
                🚨 5. Early-Warning Monitoring & Interventions
            </div>
            <p style='color: #475569; font-size: 0.90rem; line-height: 1.5;'>
                Continuous longitudinal early-warning and counseling management:
            </p>
            <ul style='color: #334155; font-size: 0.88rem; padding-left: 20px; line-height: 1.6;'>
                <li><strong>Rising Risk Trajectories:</strong> Multi-line tracking of attendance, quizzes, and simulated risk over weeks.</li>
                <li><strong>Alert Resolution Queue:</strong> Real-time triage with one-click "Mark Reviewed" updates to SQLite.</li>
                <li><strong>Intervention Outcomes Tracker:</strong> Baseline risk snapshots, follow-up evaluations, and delta metrics.</li>
                <li><strong>Live Performance Logging:</strong> Record Week 13+ data with automated ML re-scoring and alerts.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_mod6:
    st.markdown(
        """
        <div class='eduguard-card'>
            <div style='font-size: 1.15rem; font-weight: 700; color: #0f172a; margin-bottom: 8px;'>
                🔬 6. Model Explainability & Fairness Audit
            </div>
            <p style='color: #475569; font-size: 0.90rem; line-height: 1.5;'>
                Rigorous empirical validation and ethical algorithmic verification:
            </p>
            <ul style='color: #334155; font-size: 0.88rem; padding-left: 20px; line-height: 1.6;'>
                <li><strong>Scenario A vs. B Comparison:</strong> Empirical gains unlocked by mid-semester examination marks.</li>
                <li><strong>Dual Task Benchmarks:</strong> Complete regression (MAE, RMSE, R²) and classification (F1, ROC-AUC) tables.</li>
                <li><strong>Probability Calibration & ROC:</strong> Sigmoid scaling curves and decision threshold sensitivity.</li>
                <li><strong>TreeSHAP & Fairness Audits:</strong> Global feature attributions and post-hoc demographic equity slices.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")
st.caption(
    "EduGuard AI — Final Year Project | Python 3.11+, scikit-learn, XGBoost, TreeSHAP, SQLAlchemy 2.x, Streamlit, Plotly"
)
