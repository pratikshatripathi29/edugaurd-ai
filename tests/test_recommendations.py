"""
Unit Tests for EduGuard AI Advisory Recommendations Engine (Stage 9).

Validates:
1. Deterministic rule-based engine generates complete, supportive, non-punitive plans.
2. Custom threshold configurability.
3. Strict zero-PII privacy guarantee in anonymized LLM payloads.
4. Resilient Google Gemini fallback behavior when:
   - API key is missing or set to placeholder.
   - Network timeout occurs (mocked).
   - Quota limit / 429 error occurs (mocked).
   - Invalid schema / malformed JSON output occurs (mocked).
5. Pydantic schema validation integrity.
6. Interventions database persistence.
"""

from unittest.mock import MagicMock, patch
import pandas as pd
from pydantic import ValidationError
import pytest

from src.database import add_intervention, get_student_interventions
from src.recommendations import (
    DEFAULT_RECOMMENDATION_THRESHOLDS,
    GeminiPlanResponse,
    RecommendationPlan,
    build_anonymized_payload,
    generate_gemini_recommendations,
    generate_rule_based_plan,
    get_recommendations,
)


@pytest.fixture
def high_risk_profile():
    return {
        "student_id": "TEST-STU-001",
        "student_key": "GP_M_17_U_LE3_T_1_1_other_other_course_yes_no",
        "name": "Alex Student",
        "sex": "M",
        "age": 17,
        "address": "U",
        "subject": "math",
        "traveltime": 2,
        "studytime": 1,
        "failures": 2,
        "schoolsup": "no",
        "famsup": "no",
        "paid": "no",
        "higher": "yes",
        "internet": "no",
        "famrel": 3,
        "freetime": 4,
        "goout": 4,
        "Dalc": 2,
        "Walc": 3,
        "health": 2,
        "absences": 16,
        "G1": 7.0,
        "G2": 6.5,
    }


@pytest.fixture
def low_risk_profile():
    return {
        "student_id": "TEST-STU-002",
        "student_key": "MS_F_16_U_GT3_T_4_4_teacher_health_reputation_yes_yes",
        "name": "Maria Scholar",
        "sex": "F",
        "age": 16,
        "address": "U",
        "subject": "portuguese",
        "traveltime": 1,
        "studytime": 3,
        "failures": 0,
        "schoolsup": "no",
        "famsup": "yes",
        "paid": "no",
        "higher": "yes",
        "internet": "yes",
        "famrel": 5,
        "freetime": 3,
        "goout": 2,
        "Dalc": 1,
        "Walc": 1,
        "health": 5,
        "absences": 2,
        "G1": 15.0,
        "G2": 16.0,
    }


@pytest.fixture
def sample_shap_factors():
    return [
        {"feature": "G2", "shap_value": 0.45, "direction": "Elevates Risk"},
        {"feature": "absences", "shap_value": 0.32, "direction": "Elevates Risk"},
        {"feature": "higher", "shap_value": -0.18, "direction": "Mitigates Risk"},
    ]


# ==============================================================================
# 1. Rule-Based Engine Tests
# ==============================================================================
def test_rule_based_plan_high_risk(high_risk_profile, sample_shap_factors):
    """Verify rule-based plan generates complete 7-day action items for high-risk student."""
    risk_res = {"at_risk_prob": 0.82, "risk_tier": "High", "predicted_score": 6.8}
    plan = generate_rule_based_plan(high_risk_profile, risk_res, sample_shap_factors)

    assert isinstance(plan, RecommendationPlan)
    assert plan.source == "EduGuard Rule-Based Engine"
    assert plan.fallback_reason is None
    assert len(plan.daily_goals) == 7
    assert "High-Priority" in plan.weekly_plan
    assert any("Attendance Continuity" in gap for gap in plan.learning_gaps)
    assert any("Foundational Mastery" in gap for gap in plan.learning_gaps)
    assert len(plan.practice_activities) >= 2
    assert len(plan.assignment_schedule) >= 2
    assert len(plan.progress_checklist) >= 3


def test_rule_based_plan_low_risk(low_risk_profile):
    """Verify rule-based plan generates growth-oriented consolidation guidance for low-risk student."""
    risk_res = {"at_risk_prob": 0.12, "risk_tier": "Low", "predicted_score": 16.2}
    plan = generate_rule_based_plan(low_risk_profile, risk_res)

    assert isinstance(plan, RecommendationPlan)
    assert plan.source == "EduGuard Rule-Based Engine"
    assert len(plan.daily_goals) == 7
    assert "Excellence & Consolidation" in plan.weekly_plan
    assert "Spaced Repetition" in plan.revision_strategy


def test_configurable_thresholds(high_risk_profile):
    """Verify that customized thresholds alter trigger logic."""
    risk_res = {"at_risk_prob": 0.25, "risk_tier": "Low", "predicted_score": 12.0}

    # Profile with absences=8 (under default threshold 10)
    profile_moderate_abs = high_risk_profile.copy()
    profile_moderate_abs["absences"] = 8

    # Default threshold (10) -> no attendance gap
    plan_default = generate_rule_based_plan(profile_moderate_abs, risk_res)
    assert not any("Attendance Continuity" in g for g in plan_default.learning_gaps)

    # Custom strict threshold (7) -> triggers attendance gap
    plan_custom = generate_rule_based_plan(
        profile_moderate_abs,
        risk_res,
        thresholds={"high_absence_threshold": 7},
    )
    assert any("Attendance Continuity" in g for g in plan_custom.learning_gaps)


# ==============================================================================
# 2. Strict Privacy & Zero-PII Enforcement
# ==============================================================================
def test_anonymized_payload_guarantees_zero_pii(high_risk_profile, sample_shap_factors):
    """Verify that all names, student IDs, keys, and demographic fields are strictly purged."""
    risk_res = {"at_risk_prob": 0.82, "risk_tier": "High", "predicted_score": 6.8}
    payload = build_anonymized_payload(high_risk_profile, risk_res, sample_shap_factors)

    # Flatten all keys in the nested payload
    def _collect_keys(d):
        keys = []
        for k, v in d.items():
            keys.append(k)
            if isinstance(v, dict):
                keys.extend(_collect_keys(v))
        return keys

    all_keys = set(_collect_keys(payload))

    forbidden = {"student_id", "name", "student_key", "address", "sex", "age", "guardian", "famsize"}
    leakage = all_keys.intersection(forbidden)
    assert len(leakage) == 0, f"Found leaked PII keys in payload: {leakage}"

    # Verify high-level educational metrics are present
    assert payload["subject"] == "Math"
    assert payload["academic_standing"]["period_1_exam_g1"] == 7.0
    assert payload["study_and_attendance_habits"]["recorded_absences"] == 16


# ==============================================================================
# 3. Gemini Fallback Tests (Mocked API Responses)
# ==============================================================================
def test_gemini_missing_key_triggers_fallback(high_risk_profile):
    """Verify missing/placeholder API key automatically falls back to rule-based plan."""
    risk_res = {"at_risk_prob": 0.82, "risk_tier": "High", "predicted_score": 6.8}

    plan = generate_gemini_recommendations(
        profile=high_risk_profile,
        risk_result=risk_res,
        api_key="",
    )

    assert isinstance(plan, RecommendationPlan)
    assert plan.source == "EduGuard Rule-Based Engine (Fallback)"
    assert "Gemini API key is not configured" in plan.fallback_reason
    assert len(plan.daily_goals) == 7


def test_gemini_timeout_triggers_fallback(high_risk_profile):
    """Verify that a network timeout gracefully falls back to the rule-based engine."""
    risk_res = {"at_risk_prob": 0.82, "risk_tier": "High", "predicted_score": 6.8}

    with patch("google.genai.Client") as MockClient:
        mock_instance = MagicMock()
        MockClient.return_value = mock_instance
        mock_instance.models.generate_content.side_effect = TimeoutError("Deadline exceeded after 15000ms")

        plan = generate_gemini_recommendations(
            profile=high_risk_profile,
            risk_result=risk_res,
            api_key="valid-mock-key-1234567890",
        )

        assert isinstance(plan, RecommendationPlan)
        assert plan.source == "EduGuard Rule-Based Engine (Fallback)"
        assert "timed out" in plan.fallback_reason.lower()
        assert len(plan.daily_goals) == 7


def test_gemini_quota_429_triggers_fallback(high_risk_profile):
    """Verify that quota exhaustion (HTTP 429) gracefully triggers rule-based fallback."""
    risk_res = {"at_risk_prob": 0.82, "risk_tier": "High", "predicted_score": 6.8}

    with patch("google.genai.Client") as MockClient:
        mock_instance = MagicMock()
        MockClient.return_value = mock_instance
        mock_instance.models.generate_content.side_effect = Exception("429 RESOURCE_EXHAUSTED. Quota exceeded.")

        plan = generate_gemini_recommendations(
            profile=high_risk_profile,
            risk_result=risk_res,
            api_key="valid-mock-key-1234567890",
        )

        assert isinstance(plan, RecommendationPlan)
        assert plan.source == "EduGuard Rule-Based Engine (Fallback)"
        assert "429" in plan.fallback_reason
        assert len(plan.daily_goals) == 7


def test_gemini_malformed_json_triggers_fallback(high_risk_profile):
    """Verify that invalid/malformed JSON returned by LLM gracefully triggers fallback."""
    risk_res = {"at_risk_prob": 0.82, "risk_tier": "High", "predicted_score": 6.8}

    with patch("google.genai.Client") as MockClient:
        mock_instance = MagicMock()
        MockClient.return_value = mock_instance

        # Return mock response with invalid JSON text and parsed=None
        mock_resp = MagicMock()
        mock_resp.parsed = None
        mock_resp.text = "{ malformed json: missing quotes "
        mock_instance.models.generate_content.return_value = mock_resp

        plan = generate_gemini_recommendations(
            profile=high_risk_profile,
            risk_result=risk_res,
            api_key="valid-mock-key-1234567890",
        )

        assert isinstance(plan, RecommendationPlan)
        assert plan.source == "EduGuard Rule-Based Engine (Fallback)"
        assert "failed" in plan.fallback_reason.lower() or "json" in plan.fallback_reason.lower()
        assert len(plan.daily_goals) == 7


def test_gemini_successful_mocked_plan(high_risk_profile):
    """Verify successful Gemini generation parses correctly into RecommendationPlan."""
    risk_res = {"at_risk_prob": 0.82, "risk_tier": "High", "predicted_score": 6.8}

    mock_gemini_data = GeminiPlanResponse(
        weekly_plan="Intensive Math Rebuild: Review core quadratic formulas.",
        daily_goals=[f"Day {i}: Task {i}" for i in range(1, 8)],
        revision_strategy="Feynman technique and active recall.",
        practice_activities=["Solve 5 odd problems", "Formula flashcards"],
        assignment_schedule=["Day 3 draft", "Day 6 submission"],
        progress_checklist=["Checked office hours", "Solved quiz"],
        learning_gaps=["Quadratic formula mastery gap"],
        support_services=["Math Department Tutoring Clinic"],
    )

    with patch("google.genai.Client") as MockClient:
        mock_instance = MagicMock()
        MockClient.return_value = mock_instance

        mock_resp = MagicMock()
        mock_resp.parsed = mock_gemini_data
        mock_resp.text = mock_gemini_data.model_dump_json()
        mock_instance.models.generate_content.return_value = mock_resp

        plan = generate_gemini_recommendations(
            profile=high_risk_profile,
            risk_result=risk_res,
            api_key="valid-mock-key-1234567890",
            model_name="gemini-2.5-flash",
        )

        assert isinstance(plan, RecommendationPlan)
        assert plan.source == "Google Gemini (gemini-2.5-flash)"
        assert plan.fallback_reason is None
        assert plan.weekly_plan == "Intensive Math Rebuild: Review core quadratic formulas."
        assert len(plan.daily_goals) == 7


# ==============================================================================
# 4. Interventions Database Persistence Tests
# ==============================================================================
def test_save_and_retrieve_intervention():
    """Verify that an intervention plan can be committed and queried from the Interventions table."""
    # Query an existing student from the seeded database
    from src.database import Student, get_db_session
    from sqlalchemy import select

    with get_db_session() as s:
        st_obj = s.scalar(select(Student))
        assert st_obj is not None, "Seeded database must contain students."
        target_student_id = st_obj.student_id

    # Add intervention
    interv = add_intervention(
        student_id=target_student_id,
        intervention_type="ACADEMIC_REMEDIAL_PLAN",
        description="Comprehensive 7-day remedial study group in Math.",
        assigned_to="Counselor Rodriguez",
        status="Scheduled",
        data_source="TEACHER_LOGGED",
    )

    assert interv.id is not None
    assert interv.student_id == target_student_id
    assert interv.intervention_type == "ACADEMIC_REMEDIAL_PLAN"

    # Retrieve interventions
    history = get_student_interventions(target_student_id)
    assert len(history) >= 1
    assert any(h.id == interv.id for h in history)

