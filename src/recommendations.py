"""EduGuard AI - Advisory Actionable Recommendations Engine (Stage 9).

This module implements:
1. Pydantic-validated data schemas for structured pedagogical intervention plans.
2. A configurable rule-based recommendation engine mapping student profiles,
   calibrated risk categories, and real TreeSHAP attribution drivers to supportive,
   non-punitive academic plans (weekly plan, 7-day daily goals, revision strategy,
   practice activities, assignment schedule, progress checklist, support services).
3. Google Gemini integration using the official `google-genai` SDK:
   - Strictly enforces anonymization: ZERO student names, student IDs, or free-text
     comments are transmitted to external APIs.
   - Enforces structured JSON output validated against Pydantic schema.
   - Implements resilient fallbacks: On quota exhaustion, network timeout, missing keys,
     or invalid JSON, gracefully falls back to the rule-based engine and transparently
     reports data provenance (source and fallback reason).
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Union

# Ensure project root is available on sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
import pandas as pd
from pydantic import BaseModel, Field, field_validator

from src.config import RISK_PASS_MARK, RISK_THRESHOLDS

# Load environment variables
load_dotenv(_ROOT / ".env")


# ==============================================================================
# 1. Pydantic Schemas for Structured Pedagogical Guidance
# ==============================================================================
class GeminiPlanResponse(BaseModel):
    """Schema passed to Gemini enforcing structured JSON fields."""

    weekly_plan: str = Field(
        ...,
        description="High-level supportive theme and focus for the weekly recovery/improvement plan.",
    )
    daily_goals: List[str] = Field(
        ...,
        description="Concrete, actionable goals for 7 consecutive days (Day 1 through Day 7).",
    )
    revision_strategy: str = Field(
        ...,
        description="Evidence-based revision technique (e.g. active recall, spaced repetition, Cornell notes).",
    )
    practice_activities: List[str] = Field(
        ...,
        description="Actionable practice exercises, problem sets, or self-assessment tasks.",
    )
    assignment_schedule: List[str] = Field(
        ...,
        description="Milestones, drafting checkpoints, and submission deadlines for coursework.",
    )
    progress_checklist: List[str] = Field(
        ...,
        description="Verification checklist items for the student and academic advisor.",
    )
    learning_gaps: List[str] = Field(
        default_factory=list,
        description="Identified academic concepts or study habit gaps based on SHAP factors.",
    )
    support_services: List[str] = Field(
        default_factory=list,
        description="Recommended campus support services (e.g. tutoring center, counseling, office hours).",
    )


class RecommendationPlan(GeminiPlanResponse):
    """Full recommendation plan schema including metadata and provenance tracking."""

    source: str = Field(
        default="EduGuard Rule-Based Engine",
        description="Provenance of the recommendation: 'Google Gemini (<model>)' or 'EduGuard Rule-Based Engine'.",
    )
    fallback_reason: Optional[str] = Field(
        default=None,
        description="Reason why the rule-based fallback was activated, if applicable.",
    )
    anonymized_payload: Optional[Dict[str, Any]] = Field(
        default=None,
        description="The anonymized data payload evaluated by the recommendation generator.",
    )


# ==============================================================================
# 2. Configurable Rule-Based Thresholds
# ==============================================================================
DEFAULT_RECOMMENDATION_THRESHOLDS: Dict[str, Any] = {
    "high_absence_threshold": 10,        # >= 10 absences triggers attendance intervention
    "low_studytime_threshold": 2,        # <= 2 (< 5h/week) triggers study schedule boost
    "high_failures_threshold": 1,        # >= 1 prior failure triggers foundational review
    "low_exam_mark_threshold": 10.0,     # < 10.0 on G1 or G2 triggers remedial gap closure
    "high_risk_prob_threshold": 0.60,    # >= 0.60 probability triggers intensive multi-channel support
    "medium_risk_prob_threshold": 0.30,  # 0.30 - 0.60 triggers moderate proactive check-in
    "low_health_threshold": 2,           # <= 2 health status triggers wellness referral
    "low_famrel_threshold": 2,           # <= 2 family relations triggers on-campus quiet study space
    "high_alcohol_threshold": 3,         # >= 3 alcohol consumption triggers routine balance advisory
}


# ==============================================================================
# 3. Privacy & Anonymization Engine (Zero PII Enforcement)
# ==============================================================================
FORBIDDEN_PII_KEYS = {
    "student_id",
    "name",
    "student_key",
    "address",
    "guardian",
    "sex",
    "age",
    "famsize",
    "Pstatus",
    "Medu",
    "Fedu",
    "Mjob",
    "Fjob",
    "nationality",
    "phone",
    "email",
    "notes",
}


def build_anonymized_payload(
    profile: Dict[str, Any],
    risk_result: Dict[str, Any],
    top_shap_factors: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Extract and strictly anonymize student attributes for external LLM ingestion.

    CRITICAL PRIVACY GUARANTEE:
    This function explicitly filters out all student identities, names, keys,
    demographic attributes, and sensitive fields. Only high-level educational metrics
    and model attribution directions are included.

    Args:
        profile: Dictionary of student academic and behavioral fields.
        risk_result: Prediction outputs (predicted score, probability, risk tier).
        top_shap_factors: Optional list of top SHAP attribution feature dicts.

    Returns:
        Clean, anonymized dictionary safe for cloud LLM processing.

    Raises:
        ValueError: If forbidden PII keys inadvertently leak into output payload.
    """
    subject = str(profile.get("subject", "general academic")).capitalize()
    g1 = float(profile.get("G1")) if pd.notna(profile.get("G1")) else None
    g2 = float(profile.get("G2")) if pd.notna(profile.get("G2")) else None
    absences = int(profile.get("absences", 0)) if pd.notna(profile.get("absences")) else 0
    studytime = int(profile.get("studytime", 2)) if pd.notna(profile.get("studytime")) else 2
    failures = int(profile.get("failures", 0)) if pd.notna(profile.get("failures")) else 0

    study_labels = {1: "< 2 hours/week", 2: "2 - 5 hours/week", 3: "5 - 10 hours/week", 4: "> 10 hours/week"}
    study_str = study_labels.get(studytime, "2 - 5 hours/week")

    prob = float(risk_result.get("at_risk_prob", risk_result.get("prob", 0.0)))
    tier = str(risk_result.get("risk_tier", risk_result.get("risk_cat", "Moderate Risk")))
    score = float(risk_result.get("predicted_score", risk_result.get("score", 10.0)))

    # Process SHAP drivers into human-readable educational insights
    shap_drivers = []
    protective_drivers = []

    if top_shap_factors:
        for factor in top_shap_factors:
            f_name = factor.get("feature", "")
            f_dir = factor.get("direction", "")
            f_val = factor.get("shap_value", 0.0)

            if "Elevate" in f_dir or "Decreases Score" in f_dir or f_val > 0:
                shap_drivers.append(f"{f_name} (impact: {abs(f_val):.2f})")
            else:
                protective_drivers.append(f"{f_name} (protective: {abs(f_val):.2f})")

    payload = {
        "subject": subject,
        "academic_standing": {
            "period_1_exam_g1": g1,
            "period_2_exam_g2": g2,
            "exam_trend": (g2 - g1) if (g1 is not None and g2 is not None) else None,
            "passing_cutoff": RISK_PASS_MARK,
        },
        "study_and_attendance_habits": {
            "weekly_study_commitment": study_str,
            "recorded_absences": absences,
            "prior_subject_failures": failures,
        },
        "institutional_support": {
            "school_support_enrolled": profile.get("schoolsup", "no"),
            "family_support_present": profile.get("famsup", "no"),
            "extra_paid_classes": profile.get("paid", "no"),
            "higher_education_aspiration": profile.get("higher", "yes"),
        },
        "model_risk_assessment": {
            "risk_tier": tier,
            "calibrated_failure_probability": f"{prob * 100:.1f}%",
            "predicted_final_grade": f"{score:.1f} / 20",
        },
        "key_risk_factors": shap_drivers[:4],
        "key_protective_factors": protective_drivers[:3],
    }

    # Strict privacy assertion check
    def _check_keys(d: Dict[str, Any]):
        for k, v in d.items():
            if k in FORBIDDEN_PII_KEYS:
                raise ValueError(f"PRIVACY VIOLATION: Forbidden PII key '{k}' found in payload!")
            if isinstance(v, dict):
                _check_keys(v)

    _check_keys(payload)
    return payload


# ==============================================================================
# 4. Rule-Based Advisory Recommendation Engine
# ==============================================================================
def generate_rule_based_plan(
    profile: Dict[str, Any],
    risk_result: Dict[str, Any],
    top_shap_factors: Optional[List[Dict[str, Any]]] = None,
    thresholds: Optional[Dict[str, Any]] = None,
    fallback_reason: Optional[str] = None,
) -> RecommendationPlan:
    """Generate supportive, non-punitive pedagogical recommendations using deterministic rules.

    Maps profile attributes, risk categorization, and real SHAP attributions to:
    - Weekly focus plan.
    - 7-day daily goals.
    - Active revision strategy.
    - Concrete practice exercises.
    - Assignment timeline.
    - Self-monitoring checklist.
    - Institutional support referrals.

    Args:
        profile: Student demographic and academic attributes.
        risk_result: Prediction dictionary (probabilities, scores, tiers).
        top_shap_factors: Real SHAP attributions from TreeExplainer.
        thresholds: Configurable trigger thresholds. Defaults to DEFAULT_RECOMMENDATION_THRESHOLDS.
        fallback_reason: If triggered as a fallback from LLM, the error explanation.

    Returns:
        Fully populated, Pydantic-validated RecommendationPlan.
    """
    cfg = DEFAULT_RECOMMENDATION_THRESHOLDS.copy()
    if thresholds:
        cfg.update(thresholds)

    subject = str(profile.get("subject", "math")).title()
    g1 = float(profile.get("G1", 10.0)) if pd.notna(profile.get("G1")) else 10.0
    g2 = float(profile.get("G2", 10.0)) if pd.notna(profile.get("G2")) else 10.0
    absences = int(profile.get("absences", 0)) if pd.notna(profile.get("absences")) else 0
    studytime = int(profile.get("studytime", 2)) if pd.notna(profile.get("studytime")) else 2
    failures = int(profile.get("failures", 0)) if pd.notna(profile.get("failures")) else 0
    schoolsup = str(profile.get("schoolsup", "no")).lower() == "yes"

    prob = float(risk_result.get("at_risk_prob", risk_result.get("prob", 0.0)))
    tier = str(risk_result.get("risk_tier", risk_result.get("risk_cat", "Medium")))
    pred_score = float(risk_result.get("predicted_score", risk_result.get("score", 10.0)))

    learning_gaps = []
    support_services = []

    # 1. Identify Learning Gaps & Triggers
    if absences >= cfg["high_absence_threshold"]:
        learning_gaps.append(
            f"Attendance Continuity Gap: {absences} missed sessions have created sequential comprehension gaps."
        )
        support_services.append("Student Attendance Liaison: Consultation on scheduling and attendance barriers.")

    if failures >= cfg["high_failures_threshold"]:
        learning_gaps.append(
            f"Foundational Mastery Gap: Prior course setbacks in {subject} indicate prerequisite concept vulnerabilities."
        )
        support_services.append("Department Peer Tutoring: Weekly 1-on-1 foundational reinforcement.")

    if g2 < cfg["low_exam_mark_threshold"]:
        learning_gaps.append(
            f"Midterm Exam Deficit: Period 2 mark ({g2:.1f}/20) is below the {RISK_PASS_MARK} pass threshold."
        )
        support_services.append(f"{subject} Faculty Office Hours: Targeted review of recent exam problem sets.")

    if g2 < g1:
        learning_gaps.append(
            f"Negative Momentum Trend: Score declined from {g1:.1f} (G1) to {g2:.1f} (G2), indicating fatigue or syllabus escalation."
        )

    if studytime <= cfg["low_studytime_threshold"]:
        learning_gaps.append(
            "Self-Directed Study Habit Gap: Weekly study commitment is under 5 hours, limiting deep retention."
        )

    # Incorporate top SHAP factors into gaps if provided
    if top_shap_factors:
        for factor in top_shap_factors:
            f_name = factor.get("feature", "")
            f_dir = factor.get("direction", "")
            if ("Elevate" in f_dir or "Decreases" in f_dir) and f_name not in ["G1", "G2", "absences", "failures", "studytime"]:
                learning_gaps.append(
                    f"Behavioral Influence: Model attribution highlights '{f_name}' as an elevating risk factor."
                )

    if not learning_gaps:
        learning_gaps.append(
            f"Syllabus Reinforcement: Maintain consistent study routines and build resilience for upcoming final assessments in {subject}."
        )

    # Ensure baseline support service
    if not support_services:
        support_services.append(f"{subject} Academic Success Clinic: Open weekly workshop and guided study hours.")
    support_services.append("Counseling & Wellness Office: Support for test anxiety and study-life balance.")

    # 2. Build Weekly Plan Theme
    if tier == "High":
        weekly_plan = (
            f"High-Priority Stabilization Plan for {subject}: Halt academic decline by reviewing core exam errors, "
            f"instituting a daily 45-minute structured study window, and engaging departmental tutoring support."
        )
    elif tier == "Medium":
        weekly_plan = (
            f"Targeted Recovery & Mastery Plan for {subject}: Close specific concept gaps from Midterms, "
            f"establish consistent homework completion routines, and build test-taking pacing."
        )
    else:
        weekly_plan = (
            f"Excellence & Consolidation Plan for {subject}: Maintain positive momentum, deepen conceptual understanding "
            f"through advanced practice problems, and support peer study groups."
        )

    # 3. Build 7-Day Daily Goals (Strictly 7 items)
    daily_goals = [
        f"Day 1: Conduct an error analysis of the last {subject} exam; categorize mistakes into conceptual vs. calculation errors.",
        f"Day 2: Create a 1-page concept cheat sheet covering the 3 weakest topics identified on Day 1.",
        f"Day 3: Complete 45 minutes of active recall using self-made flashcards; draft questions for the teacher.",
        f"Day 4: Attend {subject} instructor office hours or tutoring session to clarify the prepared question list.",
        f"Day 5: Solve 5 intermediate practice problems under timed conditions without referring to class notes.",
        f"Day 6: Partner with a classmate for a 60-minute peer teaching session to explain core formulas aloud.",
        f"Day 7: Complete a self-assessed 30-minute mini quiz, review weekly checklist, and take planned rest.",
    ]

    # 4. Build Revision Strategy
    if tier == "High":
        revision_strategy = (
            "Pomodoro Technique with Active Problem Reproduction: Two 25-minute focus intervals daily. "
            "Instead of passive re-reading, copy problem prompts onto blank paper and reconstruct solutions from memory."
        )
    else:
        revision_strategy = (
            "Spaced Repetition & Feynman Technique: Review new concepts at 1-day, 3-day, and 7-day intervals. "
            "Explain difficult concepts in simple everyday terms to uncover hidden comprehension blind spots."
        )

    # 5. Build Practice Activities
    practice_activities = [
        f"Complete 3-5 odd-numbered practice problems from recent {subject} units, writing out step-by-step reasoning.",
        f"Construct a visual mind map connecting Period 1 and Period 2 topics to upcoming final exam objectives.",
        f"Conduct a weekly 15-minute diagnostic self-quiz using past standardized test questions.",
    ]

    # 6. Build Assignment Schedule
    assignment_schedule = [
        "Day 2: Outline key equations and problem requirements for upcoming homework assignments.",
        "Day 4: Draft problem solutions and cross-verify with peer study group or tutoring mentor.",
        "Day 6: Final proofread, rubric verification, and early submission before the deadline.",
    ]

    # 7. Build Progress Checklist
    progress_checklist = [
        "Logged 100% attendance across all scheduled class periods this week.",
        "Completed all 7 daily study goals in planner.",
        "Attended at least one faculty office hour or tutoring clinic.",
        "Completed practice exercises and logged self-quiz score.",
        "Reviewed progress with academic advisor or counselor.",
    ]

    source_label = "EduGuard Rule-Based Engine (Fallback)" if fallback_reason else "EduGuard Rule-Based Engine"

    return RecommendationPlan(
        weekly_plan=weekly_plan,
        daily_goals=daily_goals,
        revision_strategy=revision_strategy,
        practice_activities=practice_activities,
        assignment_schedule=assignment_schedule,
        progress_checklist=progress_checklist,
        learning_gaps=learning_gaps,
        support_services=support_services,
        source=source_label,
        fallback_reason=fallback_reason,
        anonymized_payload=build_anonymized_payload(profile, risk_result, top_shap_factors),
    )


# ==============================================================================
# 5. Google Gemini Integration (Official `google-genai` SDK)
# ==============================================================================
def generate_gemini_recommendations(
    profile: Dict[str, Any],
    risk_result: Dict[str, Any],
    top_shap_factors: Optional[List[Dict[str, Any]]] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    timeout_seconds: float = 15.0,
) -> RecommendationPlan:
    """Generate personalized pedagogical recommendations via Google Gemini with automated fallback.

    Privacy & Integrity Protocol:
    1. Sends ONLY anonymized educational metrics (no student name, ID, or free text).
    2. Requests strict JSON output validated against `RecommendationPlan` schema.
    3. On any failure (timeout, quota 429, missing key, network drop, schema error),
       automatically falls back to `generate_rule_based_plan` and reports the reason.

    Args:
        profile: Student demographic and academic attributes.
        risk_result: Model risk outputs (probabilities, scores, tiers).
        top_shap_factors: Real TreeSHAP attribution feature dicts.
        api_key: Optional Gemini API key (defaults to GEMINI_API_KEY from .env).
        model_name: Optional Gemini model string (defaults to GEMINI_MODEL from .env).
        timeout_seconds: Request timeout limit in seconds.

    Returns:
        RecommendationPlan with `source` reflecting Gemini or Rule-Based fallback.
    """
    # 1. Resolve credentials
    if api_key is None:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if model_name is None:
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()

    # 2. Check for missing or placeholder API key
    if not api_key or "your_gemini_api_key_here" in api_key:
        return generate_rule_based_plan(
            profile=profile,
            risk_result=risk_result,
            top_shap_factors=top_shap_factors,
            fallback_reason="Gemini API key is not configured or placeholder in .env.",
        )

    # 3. Construct strictly anonymized payload
    try:
        anonymized_payload = build_anonymized_payload(profile, risk_result, top_shap_factors)
    except Exception as e:
        return generate_rule_based_plan(
            profile=profile,
            risk_result=risk_result,
            top_shap_factors=top_shap_factors,
            fallback_reason=f"Anonymization payload error: {str(e)}",
        )

    # 4. Prepare Gemini Client and Prompt
    system_instruction = (
        "You are an empathetic, expert secondary-school pedagogical advisor and academic counselor. "
        "Your mission is to provide supportive, constructive, growth-oriented educational guidance "
        "for students experiencing academic difficulty. You receive strictly anonymized metrics and "
        "must output JSON conforming to the requested schema. "
        "STRICT GUIDELINES: "
        "- Never use punitive, discouraging, shaming, or deterministic language. "
        "- Frame all advice around skill acquisition, concept recovery, and positive study routines. "
        "- Provide concrete, actionable daily goals for 7 consecutive days (Day 1 through Day 7). "
        "- Reference the specific subject and identified risk factors in the recommendations."
    )

    prompt_content = (
        f"Based on the following anonymized student academic profile, generate a comprehensive "
        f"pedagogical intervention plan as structured JSON:\n\n"
        f"{json.dumps(anonymized_payload, indent=2)}\n\n"
        f"Return ONLY JSON matching the schema with: weekly_plan, daily_goals (exactly 7 days), "
        f"revision_strategy, practice_activities, assignment_schedule, progress_checklist, "
        f"learning_gaps, and support_services."
    )

    # 5. Execute Gemini API Call with Resilient Error Handling
    try:
        from google import genai
        from google.genai import types

        # Use v1beta endpoint for JSON structured schema support
        client = genai.Client(api_key=api_key, http_options={"api_version": "v1beta"})

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=GeminiPlanResponse,
            temperature=0.3,
            system_instruction=system_instruction,
        )

        response = client.models.generate_content(
            model=model_name,
            contents=prompt_content,
            config=config,
        )

        # Parse and validate response
        if response.parsed is not None and isinstance(response.parsed, GeminiPlanResponse):
            plan_data = response.parsed.model_dump()
            return RecommendationPlan(
                **plan_data,
                source=f"Google Gemini ({model_name})",
                fallback_reason=None,
                anonymized_payload=anonymized_payload,
            )

        # Fallback to manual text parsing if parsed attribute is raw dict or string
        raw_text = response.text or ""
        parsed_dict = json.loads(raw_text)
        validated_resp = GeminiPlanResponse.model_validate(parsed_dict)
        return RecommendationPlan(
            **validated_resp.model_dump(),
            source=f"Google Gemini ({model_name})",
            fallback_reason=None,
            anonymized_payload=anonymized_payload,
        )

    except Exception as exc:
        error_msg = str(exc)
        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            reason = f"Gemini API rate limit or quota exceeded (429): {error_msg[:120]}"
        elif "timeout" in error_msg.lower() or "deadline" in error_msg.lower():
            reason = f"Gemini API connection timed out: {error_msg[:120]}"
        elif "INVALID_ARGUMENT" in error_msg or "400" in error_msg:
            reason = f"Gemini API invalid request: {error_msg[:120]}"
        elif "ValidationError" in type(exc).__name__:
            reason = f"Gemini output failed schema validation: {error_msg[:120]}"
        else:
            reason = f"Gemini API call failed ({type(exc).__name__}): {error_msg[:120]}"

        # Seamless fallback to rule-based engine
        return generate_rule_based_plan(
            profile=profile,
            risk_result=risk_result,
            top_shap_factors=top_shap_factors,
            fallback_reason=reason,
        )


# ==============================================================================
# 6. Unified Recommendations Dispatcher
# ==============================================================================
def get_recommendations(
    profile: Dict[str, Any],
    risk_result: Dict[str, Any],
    top_shap_factors: Optional[List[Dict[str, Any]]] = None,
    use_gemini: bool = True,
) -> RecommendationPlan:
    """Unified entry point to fetch recommendations for a student profile.

    Args:
        profile: Student demographic and academic dictionary.
        risk_result: Model risk predictions dictionary.
        top_shap_factors: Optional list of top TreeSHAP factor dicts.
        use_gemini: If True, attempts Gemini generation first; falls back if unavailable.
                    If False, runs rule-based engine directly.

    Returns:
        RecommendationPlan with complete pedagogical recommendations.
    """
    if use_gemini:
        return generate_gemini_recommendations(
            profile=profile,
            risk_result=risk_result,
            top_shap_factors=top_shap_factors,
        )
    return generate_rule_based_plan(
        profile=profile,
        risk_result=risk_result,
        top_shap_factors=top_shap_factors,
    )
