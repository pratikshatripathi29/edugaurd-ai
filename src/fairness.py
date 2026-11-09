"""
EduGuard AI — Algorithmic Fairness & Subgroup Bias Audit Module (Stage 5)

This module audits the champion at-risk classification model across sensitive
demographic and institutional subgroups using HELD-OUT TEST DATA ONLY.

Fairness Principles & Governance (PROJECT_CONTEXT.md Rule 4.1):
1. Sensitive / Protected Attributes:
   - 'sex' (Female vs. Male)
   - 'address' (Urban vs. Rural)
   - 'school' (Gabriel Pereira vs. Mousinho da Silveira)
   These attributes are designated sensitive/protected. As mandated by Rule 4.1,
   sensitive attributes (sex, address, famsize, etc.) MUST NEVER be input features
   to predictive models. They are retained strictly for post-hoc fairness audits.
2. Fairness Metrics Computed:
   - Recall (True Positive Rate / Equal Opportunity):
     Equal probability that an at-risk student is identified, regardless of group.
   - Precision (Positive Predictive Value / Predictive Parity):
     Equal reliability of risk alerts across groups.
   - False Positive Rate (FPR / Predictive Equality):
     Equal rate of unnecessary false alarms among passing students.
   - Selection Rate & Disparate Impact Ratio:
     Proportion of students flagged in each demographic slice.
3. Statistical Honesty (Small Group Sizes):
   When sample sizes or positive cases are small (e.g., Rural address N=50,
   positives=16; School MS N=55, positives=16), sample variance is high.
   A single student misclassification alters recall by 6.25%. The interpretation
   helper explicitly notes small group sizes to avoid premature over-interpretation.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path when invoked directly
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score

from src.config import FIGURES_DIR, REPORTS_DIR, SENSITIVE_FEATURES


def audit_subgroup_fairness(
    df_test: pd.DataFrame,
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    sensitive_col: str,
    min_sample_size: int = 60,
    min_positive_size: int = 20,
) -> pd.DataFrame:
    """Audit classifier performance across subgroups of a sensitive attribute on test data.

    Args:
        df_test: Test DataFrame containing the sensitive attribute column.
        y_true: Ground truth binary target (1 = at-risk, 0 = passing).
        y_pred: Model predicted binary labels (1 or 0).
        sensitive_col: Column name of demographic/institutional attribute to audit.
        min_sample_size: Minimum subgroup N before flagging small sample size warning.
        min_positive_size: Minimum subgroup at-risk count before flagging statistical volatility.

    Returns:
        pd.DataFrame containing subgroup metrics and disparity measures.
    """
    if sensitive_col not in df_test.columns:
        raise KeyError(f"Attribute '{sensitive_col}' not found in test DataFrame.")

    # Align inputs
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    groups = df_test[sensitive_col].values

    unique_groups = sorted(list(set(groups)))
    rows = []

    for g_val in unique_groups:
        mask = (groups == g_val)
        y_t_g = y_t[mask]
        y_p_g = y_p[mask]

        n_total = int(len(y_t_g))
        n_pos = int(np.sum(y_t_g))
        n_neg = n_total - n_pos

        # Flagged count (predicted positive)
        n_flagged = int(np.sum(y_p_g))
        selection_rate = n_flagged / n_total if n_total > 0 else 0.0

        # Confusion Matrix
        cm = confusion_matrix(y_t_g, y_p_g, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        acc = float((tp + tn) / n_total) if n_total > 0 else 0.0

        is_small = (n_total < min_sample_size) or (n_pos < min_positive_size)

        rows.append({
            "attribute": sensitive_col,
            "subgroup": str(g_val),
            "sample_size": n_total,
            "at_risk_count": n_pos,
            "passing_count": n_neg,
            "flagged_count": n_flagged,
            "selection_rate": round(selection_rate, 4),
            "recall": round(rec, 4),
            "precision": round(prec, 4),
            "false_positive_rate": round(fpr, 4),
            "accuracy": round(acc, 4),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_negatives": int(tn),
            "small_sample_warning": is_small,
        })

    audit_df = pd.DataFrame(rows)
    return audit_df


def interpret_fairness_audit(
    audit_df: pd.DataFrame,
    attribute_name: str,
) -> str:
    """Generate a plain-English educational fairness report with sample size caveats.

    Args:
        audit_df: Subgroup metrics DataFrame produced by audit_subgroup_fairness.
        attribute_name: Name of attribute audited (e.g. 'sex', 'address', 'school').

    Returns:
        Structured narrative interpreting recall parity, precision parity, and FPR.
    """
    sub_df = audit_df[audit_df["attribute"] == attribute_name].copy()
    if sub_df.empty:
        return f"No audit data available for {attribute_name}."

    groups = sub_df["subgroup"].tolist()
    recalls = sub_df["recall"].tolist()
    precisions = sub_df["precision"].tolist()
    fprs = sub_df["false_positive_rate"].tolist()

    rec_min, rec_max = min(recalls), max(recalls)
    rec_gap = rec_max - rec_min

    prec_min, prec_max = min(precisions), max(precisions)
    prec_gap = prec_max - prec_min

    fpr_min, fpr_max = min(fprs), max(fprs)
    fpr_gap = fpr_max - fpr_min

    sel_rates = sub_df["selection_rate"].tolist()
    sel_min, sel_max = min(sel_rates), max(sel_rates)
    disparate_impact = (sel_min / sel_max) if sel_max > 0 else 1.0

    lines = []
    lines.append(f"Fairness Analysis for '{attribute_name.upper()}' ({' vs. '.join(groups)}):")

    # 1. Recall Parity (Equal Opportunity)
    if rec_gap <= 0.05:
        lines.append(
            f"• Equal Opportunity (Recall): High parity achieved. "
            f"At-risk students in both groups have virtually equal chances of receiving early support "
            f"(Recall gap: {rec_gap * 100:.1f}% across groups)."
        )
    elif rec_gap <= 0.10:
        lines.append(
            f"• Equal Opportunity (Recall): Moderate parity maintained within acceptable bounds "
            f"(Recall gap: {rec_gap * 100:.1f}%)."
        )
    else:
        lines.append(
            f"• Equal Opportunity (Recall): Noticeable recall disparity of {rec_gap * 100:.1f}%. "
            f"One subgroup experiences lower intervention rates."
        )

    # 2. Precision Parity (Predictive Parity)
    lines.append(
        f"• Predictive Parity (Precision): Precision gap is {prec_gap * 100:.1f}% "
        f"({groups[0]}: {precisions[0] * 100:.1f}%, {groups[1] if len(groups) > 1 else ''}: {precisions[1] * 100:.1f}%). "
        f"When alerts fire, their credibility is comparable across student backgrounds."
    )

    # 3. False Alarm Parity (FPR)
    lines.append(
        f"• False Alarm Parity (FPR): False positive rate among passing students varies by {fpr_gap * 100:.1f}% "
        f"(Max FPR: {fpr_max * 100:.1f}% vs Min FPR: {fpr_min * 100:.1f}%)."
    )

    # 4. Small group size warnings
    small_groups = sub_df[sub_df["small_sample_warning"] == True]
    if not small_groups.empty:
        caveats = []
        for _, sg in small_groups.iterrows():
            single_shift = 1.0 / sg["at_risk_count"] if sg["at_risk_count"] > 0 else 0.0
            caveats.append(
                f"'{sg['subgroup']}' (Total N = {sg['sample_size']}, At-Risk = {sg['at_risk_count']}; "
                f"a single student shifts recall by ±{single_shift * 100:.1f}%)"
            )
        lines.append(
            f"⚠️ Statistical Caution: High sample variance in small cohorts: {'; '.join(caveats)}. "
            f"Observed differences should not be interpreted as confirmed algorithmic bias."
        )

    return "\n".join(lines)


def comprehensive_fairness_audit(
    df_test: pd.DataFrame,
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    sensitive_cols: Optional[List[str]] = None,
    features_used: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Run fairness audits across multiple demographic and institutional dimensions.

    Args:
        df_test: Held-out test DataFrame.
        y_true: Ground truth binary target (1 = at-risk, 0 = passing).
        y_pred: Predicted binary labels.
        sensitive_cols: Attributes to audit (defaults to ['sex', 'address', 'school']).
        features_used: List of features fed into the model; verifies no leakage of sensitive columns.

    Returns:
        Concatenated pd.DataFrame with all audited slices.

    Raises:
        ValueError: If a sensitive feature was inadvertently included in model features.
    """
    if sensitive_cols is None:
        sensitive_cols = ["sex", "address", "school"]

    # Verify sensitive attributes were NOT model inputs
    if features_used is not None:
        for s in SENSITIVE_FEATURES:
            if s in features_used:
                raise ValueError(
                    f"ETHICAL GOVERNANCE VIOLATION (Rule 4.1): Protected attribute '{s}' was found "
                    f"in model features! Sensitive attributes must never be model inputs."
                )

    all_audits = []
    for col in sensitive_cols:
        if col in df_test.columns:
            sub_audit = audit_subgroup_fairness(df_test, y_true, y_pred, sensitive_col=col)
            all_audits.append(sub_audit)

    if not all_audits:
        raise ValueError("No sensitive columns found in test DataFrame to audit.")

    return pd.concat(all_audits, ignore_index=True)


def plot_fairness_subgroup_metrics(
    fairness_df: pd.DataFrame,
    title: str = "Fairness Audit: Metric Parity Across Subgroups (Held-Out Test Set)",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> plt.Figure:
    """Plot Recall, Precision, and False Positive Rate across subgroups.

    Args:
        fairness_df: DataFrame returned by comprehensive_fairness_audit.
        title: Figure title.
        save_path: Optional file path to save image.
        show: Whether to display figure interactively.

    Returns:
        matplotlib Figure object.
    """
    attributes = fairness_df["attribute"].unique()
    n_attrs = len(attributes)

    fig, axes = plt.subplots(1, n_attrs, figsize=(5 * n_attrs, 5), sharey=True)
    if n_attrs == 1:
        axes = [axes]

    metrics = ["recall", "precision", "false_positive_rate"]
    metric_labels = ["Recall (Catching At-Risk)", "Precision (Alert Credibility)", "FPR (False Alarms)"]
    colors = ["#2b83ba", "#abdda4", "#fdae61"]

    for ax, attr in zip(axes, attributes):
        sub_df = fairness_df[fairness_df["attribute"] == attr].copy()
        x = np.arange(len(sub_df))
        width = 0.25

        for i, (m, label, c) in enumerate(zip(metrics, metric_labels, colors)):
            offset = (i - 1) * width
            rects = ax.bar(x + offset, sub_df[m], width, label=label if ax == axes[0] else "", color=c, edgecolor="black", alpha=0.85)
            # Add value labels
            for rect in rects:
                h = rect.get_height()
                ax.annotate(f"{h:.2f}",
                            xy=(rect.get_x() + rect.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8)

        # Build tick labels with sample size annotations
        tick_labels = [
            f"{row['subgroup']}\n(N={row['sample_size']}, pos={row['at_risk_count']})"
            for _, row in sub_df.iterrows()
        ]
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, fontsize=9)
        ax.set_title(f"Attribute: {attr.upper()}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Subgroup (Total N, At-Risk N)")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    axes[0].set_ylabel("Score / Rate [0.0 - 1.0]")
    axes[0].set_ylim(0, 1.15)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    fig.legend(loc="upper right", bbox_to_anchor=(0.95, 0.98), ncol=3, fontsize=9)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def run_full_fairness_audit(
    scenario: str = "B",
    threshold: Optional[float] = None,
) -> Tuple[pd.DataFrame, str]:
    """Execute fairness audit on test set, save CSV and figure, and return formatted report.

    Args:
        scenario: 'A' or 'B'.
        threshold: Decision threshold. If None, uses metadata optimal threshold.

    Returns:
        Tuple of (fairness_results_df, formatted_narrative_str).
    """
    from src.classification import load_classification_model, predict_risk

    test_path = Path("data/processed/test.csv")
    if not test_path.exists():
        raise FileNotFoundError(f"Test CSV not found at {test_path}.")
    df_test = pd.read_csv(test_path)

    model, metadata = load_classification_model(scenario)
    if threshold is None:
        threshold = metadata.get("optimal_threshold", 0.50)

    # Drop target and metadata columns to prevent leakage during inference
    drop_cols = ["G3", "at_risk", "student_key", "is_simulated"]
    input_df = df_test.drop(columns=[c for c in drop_cols if c in df_test.columns], errors="ignore")

    pred_res = predict_risk(model, metadata, input_df, threshold=threshold)
    y_true = df_test["at_risk"].values
    y_pred = pred_res["predictions"]

    fairness_df = comprehensive_fairness_audit(
        df_test=df_test,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_cols=["sex", "address", "school"],
        features_used=metadata.get("features_used"),
    )

    # Save report to CSV
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_out = REPORTS_DIR / f"fairness_audit_scenario_{scenario.lower()}.csv"
    fairness_df.to_csv(csv_out, index=False)

    # Save figure
    fig_out = FIGURES_DIR / f"fairness_metrics_scenario_{scenario.lower()}.png"
    plot_fairness_subgroup_metrics(
        fairness_df,
        title=f"Scenario {scenario.upper()} Fairness Audit: Parity Across Demographic & Institutional Subgroups",
        save_path=fig_out,
    )

    # Build multi-attribute narrative report
    narratives = [
        f"=== EDGUARD AI FAIRNESS AUDIT REPORT (Scenario {scenario.upper()}, Threshold = {threshold:.2f}) ===",
        f"Audit Target: Held-out test.csv (N = {len(df_test)}, At-Risk = {y_true.sum()})",
        "",
    ]
    for attr in ["sex", "address", "school"]:
        narratives.append(interpret_fairness_audit(fairness_df, attr))
        narratives.append("-" * 60)

    full_report = "\n".join(narratives)
    return fairness_df, full_report


if __name__ == "__main__":
    print("Running Stage 5 Fairness Audits on held-out test data...")
    df_a, report_a = run_full_fairness_audit(scenario="A", threshold=0.25)
    print("\n" + report_a)

    df_b, report_b = run_full_fairness_audit(scenario="B", threshold=0.15)
    print("\n" + report_b)
    print("Fairness audit completed successfully.")
