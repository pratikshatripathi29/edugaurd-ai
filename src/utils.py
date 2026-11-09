"""
EduGuard AI — Helper Utilities and UI Styling System (Stage 7)

Provides:
- CSS injector for custom Streamlit styling
- Reusable HTML card builders for KPIs, risk badges, and ethical governance notices
- Unified Plotly styling aligned with our navy/white/teal visual identity
"""

from pathlib import Path
from typing import Optional
import plotly.graph_objects as go
import streamlit as st

_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
_STYLES_CSS = _ASSETS_DIR / "styles.css"


def load_custom_css() -> None:
    """Inject assets/styles.css into the active Streamlit app."""
    if _STYLES_CSS.exists():
        with open(_STYLES_CSS, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def get_kpi_html(
    title: str,
    value: str,
    subtitle: Optional[str] = None,
    border_color: str = "#0d9488",
) -> str:
    """Generate a clean HTML KPI metric box.

    Args:
        title: Uppercase label.
        value: Formatted numerical value.
        subtitle: Optional descriptive subtext.
        border_color: Top accent color (default teal).

    Returns:
        HTML string.
    """
    sub_html = f"<div class='kpi-subtitle'>{subtitle}</div>" if subtitle else ""
    return f"""
    <div class='kpi-container' style='border-top-color: {border_color};'>
        <div class='kpi-title'>{title}</div>
        <div class='kpi-value'>{value}</div>
        {sub_html}
    </div>
    """


def get_risk_badge_html(risk_category: str) -> str:
    """Return colored pill badge HTML for Low, Medium, or High risk."""
    clean_cat = risk_category.strip().capitalize()
    if clean_cat == "High":
        return "<span class='badge-high'>High Risk</span>"
    elif clean_cat in {"Medium", "Moderate"}:
        return "<span class='badge-medium'>Medium Risk</span>"
    return "<span class='badge-low'>Low Risk</span>"


def get_simulated_tag_html() -> str:
    """Return purple SIMULATED provenance pill tag."""
    return "<span class='badge-simulated'>SIMULATED</span>"


def get_advisory_notice_html() -> str:
    """Generate the standardized ethical decision-support notice."""
    return """
    <div class='advisory-notice'>
        <div class='advisory-title'>
            <span>🛡️</span> Responsible AI Advisory Decision-Support Notice
        </div>
        <p class='advisory-text'>
            EduGuard AI provides <strong>advisory decision-support, not automated decision-making</strong>.
            All risk categories, probabilities, and scores are statistical indicators to help educators
            allocate proactive mentoring. Predictions must never be used for punitive actions, automated
            tracking, or deterministic course exclusions.
        </p>
    </div>
    """


def get_simulated_data_notice_html() -> str:
    """Generate the standardized synthetic data provenance notice (Rule 2.1)."""
    return """
    <div class='simulated-notice'>
        <div style='font-weight: 700; color: #4c1d95; font-size: 0.90rem; margin-bottom: 4px;'>
            <span>⚠️</span> Data Provenance Notice (Rule 2.1 — SIMULATED)
        </div>
        <p style='font-size: 0.84rem; color: #5b21b6; margin: 0;'>
            Student demographic attributes and baseline exam grades are drawn from the real UCI Student Performance
            dataset. However, <strong>all longitudinal weekly tracking logs (Weeks 1–12)</strong> are
            <strong>synthetically simulated</strong> for alert testing and demonstration.
        </p>
    </div>
    """


def apply_plotly_theme(fig: go.Figure, title: Optional[str] = None) -> go.Figure:
    """Apply consistent EduGuard AI navy/white/teal aesthetics to a Plotly figure."""
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>" if title else "",
            font=dict(size=14, color="#0f172a", family="Segoe UI"),
            x=0.02,
        ),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(family="Segoe UI", color="#334155", size=11),
        margin=dict(l=40, r=30, t=50, b=40),
        xaxis=dict(
            showgrid=True,
            gridcolor="#f1f5f9",
            zeroline=False,
            linecolor="#cbd5e1",
            tickfont=dict(color="#475569"),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#f1f5f9",
            zeroline=False,
            linecolor="#cbd5e1",
            tickfont=dict(color="#475569"),
        ),
        legend=dict(
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="#e2e8f0",
            borderwidth=1,
            font=dict(size=10),
        ),
    )
    return fig
