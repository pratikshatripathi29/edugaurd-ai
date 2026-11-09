# EduGuard AI — Project Context & Engineering Governance Rules

> **Status:** Active Reference Document (Stage 0 Baseline)  
> **Target Audience:** Developers, Mentors, Evaluators, and Future AI Coding Sessions  
> **Stack:** Python 3.11+, pandas, scikit-learn, XGBoost, SHAP, Streamlit, Plotly, SQLAlchemy/SQLite, Google Gemini (Optional)

---

## 1. System Mission & Ethical Stance
EduGuard AI is a responsible machine learning decision-support system designed to detect students at academic risk, explain key risk factors using SHAP (SHapley Additive exPlanations), provide actionable pedagogical recommendations, and monitor student trajectories over time.

### Fundamental Tenets:
1. **Advisory Decision Support, Not Autonomous Decisions:**
   - Predictions and risk scores are advisory tools to empower human teachers and academic advisors.
   - Predictions must **never** be used for punitive measures, automated tracking, or deterministic actions.
2. **Operational Definition of Academic Risk:**
   - Academic risk is defined operationally as a final grade $G3 < 10$ on the Portuguese $0 - 20$ grading scale.
   - This is an **operational convention** for triage and early warning, *not* an intrinsic or universal label of a student's intelligence or potential.

---

## 2. Data Governance & Integrity Rules

### Rule 2.1 — Real Data vs. Simulated Data
- **Real Data:**
  - Source: UCI Student Performance Dataset (`student-mat.csv` for Mathematics, `student-por.csv` for Portuguese).
  - Format: Semicolon-delimited (`sep=';'`).
  - Storage: `data/raw/` (ignored in version control).
- **Simulated Data:**
  - Any synthetic profiles, weekly logs, or stress-test cases must reside in a separate file (e.g., `data/processed/simulated_students.csv`).
  - **Mandatory Labeling:** Simulated data MUST be explicitly labelled **`SIMULATED`** in code comments, database tables, UI dashboard cards, and report exports.
  - Simulated data must never be mixed with or pollute real UCI test evaluation splits.

### Rule 2.2 — Duplicate Students & Grouped Cross-Validation
- As documented by Cortez & Silva (2008), approximately **382 students** appear in both `student-mat.csv` and `student-por.csv`.
- A composite key (`student_key`) must be constructed from student-identifying attributes:
  `['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'nursery', 'internet']`.
- **Leakage Prevention Split:** All cross-validation and train/test splits must use **`GroupKFold`** or `GroupShuffleSplit` on `student_key`. The same student must **never** appear in both the training set and the test set across subjects.

---

## 3. Strict Machine Learning Rules & Zero-Leakage Policy

### Rule 3.1 — Target Leakage
- **Never use $G3$ as an input to predict $G3$:**
  $G3$ is the target variable. Any derived version of $G3$ (e.g., `passed`, `at_risk`) must also be completely excluded from training inputs.

### Rule 3.2 — Preprocessing Fit
- All transformers (imputers, one-hot encoders, standardizers) must be fitted **strictly on the training split only** (`fit` or `fit_transform` on `X_train`), then applied to test/validation splits (`transform` on `X_test`).
- Preprocessing steps must be encapsulated inside **`sklearn.pipeline.Pipeline`** or **`ColumnTransformer`** objects to guarantee zero leakage.

### Rule 3.3 — Modeling Scenarios
Every model experiment and evaluation must be explicitly reported under two distinct pedagogical scenarios:
- **Scenario A (Early-Warning / Early-Semester):**
  - Grades $G1$ (Period 1) and $G2$ (Period 2) are **not available**.
  - Models predict $G3$ and risk using only historical, demographic, and early-behavioral features.
- **Scenario B (Mid-Semester Update):**
  - Period grades $G1$ and $G2$ **are available** and included as predictive features.
  - Reflects the updated risk assessment after preliminary exams.

---

## 4. Algorithmic Fairness & Protected Attributes

### Rule 4.1 — Sensitive Attributes are NOT Predictive Features
The following features are designated sensitive/protected attributes:
- `sex`, `age`, `address`, `famsize`, `Pstatus`, `Medu`, `Fedu`, `Mjob`, `Fjob`, `guardian`, and any nationality/citizenship fields.
- **Strict Prohibition:** These attributes must **NEVER** be input features into the predictive regression or classification models.
- **Permitted Use:** These attributes are retained in separate metadata splits **strictly for post-hoc fairness audits** (e.g., assessing Demographic Parity, Disparate Impact, and Equalized Odds across demographic subgroups).

---

## 5. Scientific Honesty & Metrics Protocol
- **Never invent or extrapolate metrics:**
  - Never fabricate test scores, accuracy figures, RMSE, or SHAP importance values.
  - Every metric quoted in notebooks, markdown reports, dashboard cards, or prompt replies must be directly computed by executed Python code.
- Report both regression metrics ($RMSE$, $MAE$, $R^2$, $MedAE$) and classification metrics ($ROC\text{-}AUC$, $PR\text{-}AUC$, $F1\text{-score}$, $Precision$, $Recall$, $Brier\ Score$).

---

## 6. Project Architecture Overview

| Path | Purpose |
| :--- | :--- |
| `data/raw/` | Unmodified UCI raw CSV files (`student-mat.csv`, `student-por.csv`). |
| `data/processed/` | Cleaned, validated, merged, and split datasets. |
| `notebooks/` | Interactive Jupyter notebooks for EDA, prototyping, and fairness evaluation. |
| `models/` | Serialized scikit-learn / XGBoost pipeline artifacts (`.joblib`). |
| `reports/figures/` | Saved high-resolution figures, SHAP plots, and evaluation charts. |
| `src/config.py` | Centralized paths, random seeds, risk cutoffs, and feature governance lists. |
| `src/data_loader.py` | Semicolon parsing, schema validation, and student key linking. |
| `src/preprocessing.py` | Leak-free `Pipeline` and `ColumnTransformer` definitions. |
| `src/features.py` | Feature extraction for Scenario A and Scenario B. |
| `src/regression.py` | Final grade $G3$ regression estimators and evaluation. |
| `src/classification.py` | Operational at-risk ($G3 < 10$) classifiers and calibration. |
| `src/explainability.py` | TreeSHAP/LinearSHAP global and local attribution tools. |
| `src/fairness.py` | Algorithmic fairness audits across sensitive demographic slices. |
| `src/simulation.py` | Synthetic profile generation (strictly tagged `SIMULATED`). |
| `src/recommendations.py` | Rule-based and optional Gemini LLM advisory recommendation cards. |
| `src/monitoring.py` | Longitudinal weekly tracking and early-warning escalation logic. |
| `src/database.py` | SQLAlchemy models and SQLite persistence layer. |
| `src/utils.py` | Reusable formatting, metric reporting, and visual styling utilities. |
| `pages/` | Multi-page Streamlit views (e.g., student lookup, cohort monitor, fairness audit). |
| `assets/` | Static project assets (diagrams, logos, CSS). |
| `tests/` | Pytest test suite covering data integrity, pipelines, and fairness guarantees. |
| `app.py` | Streamlit entrypoint dashboard. |
| `requirements.txt` | Pinned compatible package dependencies. |
| `.env.example` | Template for environment variables (Gemini API credentials). |
| `.gitignore` | Prevents committing `.venv`, raw data, bytecode, and model binaries. |

