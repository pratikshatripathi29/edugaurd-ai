# 🎓 EduGuard AI: Student Performance Prediction & Early-Warning System

[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](https://streamlit.io/)
[![SHAP](https://img.shields.io/badge/XAI-SHAP-green.svg)](https://shap.readthedocs.io/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost%20%7C%20Scikit--Learn-orange.svg)](https://xgboost.readthedocs.io/)

**EduGuard AI** is a responsible, explainable machine learning platform engineered to help educators and academic mentors identify students at risk of academic failure early enough to provide meaningful, personalized interventions.

Built with **Python**, **scikit-learn**, **XGBoost**, **SHAP**, **Plotly**, **Streamlit**, and **SQLAlchemy/SQLite**, the system bridges the gap between predictive accuracy and pedagogical accountability.

---

## 🌟 Core Pillars

1. **Advisory Decision Support:**  
   EduGuard AI provides decision *support*, not decisions. All predictions and alerts are advisory inputs intended to assist human educators, never to automate punitive actions.
2. **Strict Algorithmic Fairness:**  
   Demographic and socioeconomic traits (`sex`, `age`, `address`, `famsize`, `Pstatus`, `Medu`, `Fedu`, `Mjob`, `Fjob`, `guardian`, `nationality`) are strictly excluded from predictive model inputs and reserved solely for post-hoc fairness audits.
3. **Transparent Explainability:**  
   Every risk classification is supported by SHAP attributions, highlighting actionable factors (e.g., study time, absences, past academic performance) that can guide realistic pedagogical interventions.
4. **Dual-Horizon Evaluation:**  
   - **Scenario A (Early Warning):** Predictions before Period 1 ($G1$) and Period 2 ($G2$) exam grades exist.
   - **Scenario B (Mid-Semester Update):** Updated predictions once $G1$ and $G2$ grades are recorded.
5. **Zero Data Leakage:**  
   Leak-free `sklearn.pipeline.Pipeline` workflows with Grouped Cross-Validation (`GroupKFold`) linked on `student_key` to account for students enrolled in both Mathematics and Portuguese courses.

---

## 📁 Repository Structure

```text
eduguard-ai/
│
├── data/
│   ├── raw/                 # Original UCI Student Performance datasets (student-mat.csv, student-por.csv)
│   └── processed/           # Processed, validated, and split datasets (including SIMULATED records)
│
├── notebooks/               # Jupyter notebooks for EDA, prototyping, and fairness experiments
├── models/                  # Saved scikit-learn & XGBoost model pipeline checkpoints (.joblib)
├── reports/
│   └── figures/             # Exported evaluation figures, confusion matrices, and SHAP plots
│
├── src/                     # Modular Python source package
│   ├── __init__.py          # Package initialization
│   ├── config.py            # Central configuration: paths, random state, risk marks, feature lists
│   ├── data_loader.py       # Raw data ingestion, semicolon parsing, and student key linking
│   ├── preprocessing.py     # Leak-free sklearn Pipelines and ColumnTransformers
│   ├── features.py          # Feature extraction for Scenario A and Scenario B
│   ├── regression.py        # Final grade (G3) regression models and evaluation
│   ├── classification.py    # At-risk (G3 < 10) classification models and calibration
│   ├── explainability.py    # TreeSHAP & LinearSHAP global/local attributions
│   ├── fairness.py          # Algorithmic fairness audit across demographic slices
│   ├── simulation.py        # Synthetic student trajectory generation (strictly tagged SIMULATED)
│   ├── recommendations.py   # Rule-based and optional Gemini LLM advisory recommendation cards
│   ├── monitoring.py        # Longitudinal weekly tracking and tiered escalation logic
│   ├── database.py          # SQLAlchemy models and SQLite persistence layer
│   └── utils.py             # Reusable formatting, metrics reporting, and visual utilities
│
├── pages/                   # Streamlit multi-page dashboard views
├── assets/                  # Images, UI styling, and static documentation assets
├── tests/                   # Pytest test suite for data validation and pipeline integrity
│   └── __init__.py
│
├── app.py                   # Streamlit main dashboard entrypoint
├── requirements.txt         # Pinned compatible dependency specifications
├── .env.example             # Example environment variable file for API keys
├── .gitignore               # Git ignore rules for environments, raw data, caches, and models
├── PROJECT_CONTEXT.md       # Immutable engineering guidelines & project rules
└── README.md                # Project documentation and quickstart guide
```

---

## 🚀 Quickstart Guide (Windows + PowerShell)

### 1. Prerequisites
Ensure you have **Python 3.11+** installed on your system.

### 2. Activate Virtual Environment
Open PowerShell inside `D:\eduguard-ai`:

```powershell
# Set execution policy for the current session (if not already permitted)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Verify Environment Installation
Verify that all core libraries import cleanly:
```powershell
python -c "import pandas, numpy, sklearn, xgboost, shap, streamlit, plotly, sqlalchemy, dotenv, google.genai; print('All EduGuard AI dependencies verified successfully!')"
```

### 5. Launch the Dashboard
```powershell
streamlit run app.py
```

---

## 🧭 Project Roadmap

- [x] **Stage 0: Project Scaffolding & Setup** (Current)
  - Directory layout, pinned dependencies, central config, and governance rules.
- [ ] **Stage 1: Ingestion, EDA & Grouped Preprocessing**
  - Dataset loading, schema validation, student linking, leakage-free pipelines.
- [ ] **Stage 2: Model Training & Evaluation (Regression & Classification)**
  - Dual scenarios (Scenario A vs. Scenario B), GroupKFold validation, metrics reporting.
- [ ] **Stage 3: Explainability (SHAP) & Algorithmic Fairness Audit**
  - Waterfall/force plots, demographic parity, disparate impact analysis.
- [ ] **Stage 4: Advisory Recommendations, Longitudinal Monitoring & Database**
  - Intervention cards, SQLite audit logging, longitudinal early warnings.
- [ ] **Stage 5: Interactive Streamlit Multi-Page Application**
  - Full UI dashboard for teachers, mentors, and administrators.

---

## ⚖️ License & Ethical Declaration
This software is developed strictly for educational decision-support. Predictions should always be cross-referenced with educator observations.
Licensed under the [MIT License](LICENSE).

