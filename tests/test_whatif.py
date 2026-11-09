"""
Unit Tests for EduGuard AI What-If Scenario Simulator (Stage 8)

Validates that:
1. Unchanged inputs yield deterministically identical outputs.
2. Modified inputs produce valid, bounded predictions (score in [0, 20], prob in [0, 1]).
3. Feature adjustments shift risk probabilities and predicted scores logically.
4. Engineered features (e.g. grade_trend, alcohol_total) update consistently.
5. Target leakage columns are strictly rejected.
"""

import numpy as np
import pandas as pd
import pytest

from src.classification import load_classification_model, predict_risk
from src.features import add_engineered_features
from src.regression import load_model, predict_score


@pytest.fixture(scope="module")
def models_scenario_b():
    """Load Scenario B models."""
    clf, meta_clf = load_classification_model("B")
    reg, meta_reg = load_model("B")
    return clf, meta_clf, reg, meta_reg


@pytest.fixture(scope="module")
def models_scenario_a():
    """Load Scenario A models."""
    clf, meta_clf = load_classification_model("A")
    reg, meta_reg = load_model("A")
    return clf, meta_clf, reg, meta_reg


@pytest.fixture
def base_student_b():
    """Provide a standard baseline student dictionary for Scenario B."""
    return {
        "school": "GP",
        "subject": "math",
        "reason": "course",
        "traveltime": 1,
        "studytime": 2,
        "failures": 1,
        "schoolsup": "no",
        "famsup": "yes",
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
        "absences": 8,
        "G1": 9.0,
        "G2": 9.5,
    }


def test_unchanged_inputs_give_identical_outputs_scenario_b(models_scenario_b, base_student_b):
    """Verify that evaluating the exact same input twice yields bit-for-bit identical outputs."""
    clf, meta_clf, reg, meta_reg = models_scenario_b

    df1 = add_engineered_features(pd.DataFrame([base_student_b]), include_trend=True)
    df2 = add_engineered_features(pd.DataFrame([base_student_b]), include_trend=True)

    # First evaluation
    prob1 = predict_risk(clf, meta_clf, df1, threshold=0.15)
    score1 = predict_score(reg, meta_reg, df1)

    # Second evaluation
    prob2 = predict_risk(clf, meta_clf, df2, threshold=0.15)
    score2 = predict_score(reg, meta_reg, df2)

    assert np.isclose(prob1["probabilities"][0], prob2["probabilities"][0]), "Probabilities must be identical."
    assert np.isclose(score1[0], score2[0]), "Scores must be identical."
    assert prob1["risk_tiers"][0] == prob2["risk_tiers"][0], "Risk categories must match."
    assert prob1["predictions"][0] == prob2["predictions"][0], "Binary flags must match."


def test_unchanged_inputs_give_identical_outputs_scenario_a(models_scenario_a, base_student_b):
    """Verify determinism under Scenario A (no G1/G2)."""
    clf, meta_clf, reg, meta_reg = models_scenario_a

    # Remove G1/G2 for Scenario A
    student_a = {k: v for k, v in base_student_b.items() if k not in ["G1", "G2"]}
    df1 = add_engineered_features(pd.DataFrame([student_a]), include_trend=False)
    df2 = add_engineered_features(pd.DataFrame([student_a]), include_trend=False)

    prob1 = predict_risk(clf, meta_clf, df1, threshold=0.25)
    score1 = predict_score(reg, meta_reg, df1)

    prob2 = predict_risk(clf, meta_clf, df2, threshold=0.25)
    score2 = predict_score(reg, meta_reg, df2)

    assert np.isclose(prob1["probabilities"][0], prob2["probabilities"][0])
    assert np.isclose(score1[0], score2[0])
    assert prob1["risk_tiers"][0] == prob2["risk_tiers"][0]


def test_changed_inputs_produce_valid_bounded_outputs(models_scenario_b, base_student_b):
    """Verify that sweeping inputs across ranges produces valid, bounded probabilities and scores."""
    clf, meta_clf, reg, meta_reg = models_scenario_b

    # Test extreme feature combinations
    test_cases = [
        {"G1": 0.0, "G2": 0.0, "absences": 45, "failures": 3, "studytime": 1},  # Extreme struggling
        {"G1": 20.0, "G2": 20.0, "absences": 0, "failures": 0, "studytime": 4}, # Flawless attainment
        {"G1": 8.0, "G2": 15.0, "absences": 12, "failures": 2, "studytime": 3}, # Strong recovery
        {"G1": 15.0, "G2": 6.0, "absences": 22, "failures": 0, "studytime": 1}, # Sudden decline
    ]

    for case in test_cases:
        modified_dict = base_student_b.copy()
        modified_dict.update(case)

        df_input = add_engineered_features(pd.DataFrame([modified_dict]), include_trend=True)
        prob_res = predict_risk(clf, meta_clf, df_input, threshold=0.15)
        score_res = predict_score(reg, meta_reg, df_input)

        p = prob_res["probabilities"][0]
        s = score_res[0]
        cat = prob_res["risk_tiers"][0]

        # Valid domains
        assert 0.0 <= p <= 1.0, f"Probability {p} out of bounds [0, 1] for case {case}"
        assert 0.0 <= s <= 20.0, f"Score {s} out of bounds [0, 20] for case {case}"
        assert cat in {"Low", "Medium", "High"}, f"Invalid risk category: {cat}"


def test_academic_momentum_positive_shift(models_scenario_b, base_student_b):
    """Verify that improving exam grades (G1, G2) increases predicted score and reduces risk."""
    clf, meta_clf, reg, meta_reg = models_scenario_b

    # Failing baseline
    failing_student = base_student_b.copy()
    failing_student["G1"] = 5.0
    failing_student["G2"] = 6.0
    df_failing = add_engineered_features(pd.DataFrame([failing_student]), include_trend=True)

    prob_failing = predict_risk(clf, meta_clf, df_failing, threshold=0.15)["probabilities"][0]
    score_failing = predict_score(reg, meta_reg, df_failing)[0]

    # Improved hypothetical
    improved_student = base_student_b.copy()
    improved_student["G1"] = 14.0
    improved_student["G2"] = 16.0
    df_improved = add_engineered_features(pd.DataFrame([improved_student]), include_trend=True)

    prob_improved = predict_risk(clf, meta_clf, df_improved, threshold=0.15)["probabilities"][0]
    score_improved = predict_score(reg, meta_reg, df_improved)[0]

    assert score_improved > score_failing, "Improving marks must increase predicted final score."
    assert prob_improved < prob_failing, "Improving marks must decrease at-risk probability."
    assert df_improved["grade_trend"].iloc[0] == 2.0, "grade_trend must equal G2 - G1."


def test_reducing_absences_and_failures_lowers_risk_scenario_a(models_scenario_a, base_student_b):
    """Verify that in Scenario A, lower absences and 0 failures produce lower risk than high absences/failures."""
    clf, meta_clf, reg, meta_reg = models_scenario_a

    # High risk profile
    high_risk_a = {k: v for k, v in base_student_b.items() if k not in ["G1", "G2"]}
    high_risk_a["failures"] = 3
    high_risk_a["absences"] = 30
    high_risk_a["studytime"] = 1
    df_high = add_engineered_features(pd.DataFrame([high_risk_a]), include_trend=False)

    p_high = predict_risk(clf, meta_clf, df_high, threshold=0.25)["probabilities"][0]
    s_high = predict_score(reg, meta_reg, df_high)[0]

    # Low risk profile
    low_risk_a = {k: v for k, v in base_student_b.items() if k not in ["G1", "G2"]}
    low_risk_a["failures"] = 0
    low_risk_a["absences"] = 2
    low_risk_a["studytime"] = 3
    df_low = add_engineered_features(pd.DataFrame([low_risk_a]), include_trend=False)

    p_low = predict_risk(clf, meta_clf, df_low, threshold=0.25)["probabilities"][0]
    s_low = predict_score(reg, meta_reg, df_low)[0]

    assert p_low < p_high, "Low absences and zero failures must yield lower risk probability."
    assert s_low > s_high, "Low absences and zero failures must yield higher predicted score."


def test_whatif_rejects_target_leakage(models_scenario_b, base_student_b):
    """Verify that passing target leakage columns (G3, at_risk) raises a ValueError."""
    clf, meta_clf, _, _ = models_scenario_b

    leaky_student = base_student_b.copy()
    leaky_student["G3"] = 12.0  # Illegal target leakage

    df_leaky = add_engineered_features(pd.DataFrame([leaky_student]), include_trend=True)

    with pytest.raises(ValueError, match="DATA LEAKAGE ERROR"):
        predict_risk(clf, meta_clf, df_leaky, threshold=0.15)

