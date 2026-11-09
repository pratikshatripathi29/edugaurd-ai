# 🎓 EduGuard AI: Student Performance Prediction & Early-Warning System

[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost%20%7C%20Scikit--Learn-orange.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/XAI-TreeSHAP-green.svg)](https://shap.readthedocs.io/)
[![Tests: 80 Passed](https://img.shields.io/badge/Tests-80%20Passed-brightgreen.svg)](tests/)

> 🌐 **Live Web Application:** [EduGuard AI Live App](https://your-eduguard-ai-app-url.streamlit.app) *(Link placeholder)*

---

## 📌 Project Description

**EduGuard AI** is a responsible, explainable machine learning decision-support platform built to detect students at academic risk early, explain risk drivers with game-theoretic interpretability (TreeSHAP), and guide educators with actionable, personalized intervention plans.

Trained on the [UCI Student Performance Dataset](https://archive.ics.uci.edu/dataset/320/student+performance) (Cortez & Silva, 2008), the system predicts final examination marks ($G3 \in [0, 20]$) and operational at-risk status ($G3 < 10$) across two academic horizons:
- **Scenario A (Early-Warning):** Early-semester triage before Period 1 ($G1$) and Period 2 ($G2$) exam marks exist.
- **Scenario B (Mid-Semester):** Updated risk forecasts once interim marks ($G1$ and $G2$) are recorded.

---

## ⚖️ Ethical Governance & Decision-Support Disclaimer

> [!IMPORTANT]
> **Advisory Decision Support, Not Autonomous Decision-Making:**  
> All model predictions, risk tiers (Low, Medium, High), and early-warning alerts produced by EduGuard AI are intended **solely to assist educators, mentors, and academic counselors**. They must **never** be used to automate punitive actions, deny educational opportunities, or replace professional human pedagogical judgment.

---

## 📊 Provenance Notice: Real vs. SIMULATED Data

- **Real Observation Data:** Baseline demographics, school context, study habits, absences, and exam marks ($G1, G2, G3$) originate from the real UCI Student Performance dataset (Mathematics and Portuguese cohorts).
- **SIMULATED Longitudinal Data:** All multi-week time-series records (weekly attendance rates, quiz scores, assignment completion, study hours over Weeks 1–12, and trajectory labels) are **synthetically simulated** for demonstration and early-warning escalation testing. Every weekly database record and dashboard chart explicitly carries the provenance flag **`SIMULATED`**.

---

## 🚀 Key Features

1. **Dual-Horizon Modeling:**
   - Evaluates models with and without interim exam marks to quantify the value of mid-semester academic momentum.
2. **Regression & Calibrated Classification:**
   - Final score prediction via Ridge, Random Forest, and XGBoost inside leak-free pipelines.
   - At-risk probability estimation calibrated via sigmoid scaling with threshold-analysis trade-offs.
3. **TreeSHAP Explainability:**
   - Global feature importance bars, beeswarm summary plots, and local student waterfall attributions.
   - Generates plain-English diagnostic explanations directly from empirical Shapley values.
4. **Algorithmic Fairness Audits:**
   - Sensitive attributes (`sex`, `address`, `school`, `age`, parental status) are **strictly excluded** from model training.
   - Post-hoc audits evaluate Demographic Parity, Recall, and False Positive Rates across subgroups.
5. **Interactive What-If Simulator:**
   - Explore sensitivity to changes in study time, absences, and exam marks with explicit non-causal advisory disclaimers.
6. **AI Pedagogical Recommendations:**
   - Structured 7-day recovery plans powered by **Google Gemini** (structured JSON schema via the official `google-genai` SDK) with automatic fallback to a deterministic rule-based engine.
   - Zero Student PII sent to external APIs.
7. **Longitudinal Early-Warning & Intervention Tracker:**
   - Multi-week trajectory tracking, alert triage with "Mark Reviewed" SQLite status updates, and intervention baseline vs. follow-up outcome tracking.

---

## 📈 Empirical Evaluation Results (From `reports/`)

All values below are parsed directly from experimental evaluation reports in [`reports/`](reports/) on the held-out test split ($n=203$). Zero metrics are invented.

### 1. Final Grade ($G3$) Regression Performance

| Scenario | Model | Test MAE | Test RMSE | Test $R^2$ | Non-Zero RMSE ($G3 > 0$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Scenario A** *(Early)* | Baseline (Mean) | 2.722 | 3.692 | -0.0002 | 2.879 |
| **Scenario A** *(Early)* | Ridge Regression | 2.291 | 3.055 | 0.3154 | 2.479 |
| **Scenario A** *(Early)* | Random Forest | 2.064 | 2.765 | 0.4391 | 2.512 |
| **Scenario A** *(Early)* | **XGBoost Regressor** | 2.154 | 2.807 | 0.4220 | 2.520 |
| **Scenario B** *(Mid-Sem)* | Baseline (Mean) | 2.722 | 3.692 | -0.0002 | 2.879 |
| **Scenario B** *(Mid-Sem)* | Ridge Regression | 0.887 | 1.350 | 0.8663 | 0.978 |
| **Scenario B** *(Mid-Sem)* | Random Forest | 0.824 | 1.270 | 0.8816 | 1.123 |
| **Scenario B** *(Mid-Sem)* | **XGBoost Regressor** | **0.821** | **1.260** | **0.8836** | **1.132** |

### 2. At-Risk ($G3 < 10$) Classification & Calibration

| Scenario | Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Brier Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Scenario A** | Logistic Regression | 79.31% | 65.00% | 27.08% | 0.382 | 0.818 | 0.573 | 0.1404 |
| **Scenario A** | Decision Tree | 76.35% | 0.00% | 0.00% | 0.000 | 0.709 | 0.417 | 0.1614 |
| **Scenario A** | **Random Forest** | **80.79%** | **66.67%** | **37.50%** | **0.480** | **0.831** | **0.597** | **0.1345** |
| **Scenario A** | XGBoost Classifier | 79.80% | 68.42% | 27.08% | 0.388 | 0.809 | 0.581 | 0.1394 |
| **Scenario B** | Logistic Regression | 87.68% | 76.74% | 68.75% | 0.725 | 0.956 | 0.865 | 0.0731 |
| **Scenario B** | Decision Tree | 85.71% | 67.27% | 77.08% | 0.718 | 0.851 | 0.642 | 0.1136 |
| **Scenario B** | **Random Forest** | **90.15%** | **80.43%** | **77.08%** | **0.787** | **0.963** | **0.896** | **0.0679** |
| **Scenario B** | XGBoost Classifier | 88.67% | 75.51% | 77.08% | 0.763 | 0.951 | 0.876 | 0.0835 |

### 3. Subgroup Demographic Fairness Audit (Scenario B)

| Attribute | Subgroup | Sample ($n$) | Selection Rate | Recall | Precision | False Positive Rate |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Sex** | Female (`F`) | 119 | 30.25% | **92.86%** | 72.22% | 10.99% |
| **Sex** | Male (`M`) | 84 | 33.33% | **90.00%** | 64.29% | 15.62% |
| **Address** | Rural (`R`)* | 50 | 42.00% | **100.00%** | 76.19% | 14.71% |
| **Address** | Urban (`U`) | 153 | 28.10% | **87.50%** | 65.12% | 12.40% |
| **School** | Gabriel Pereira (`GP`) | 148 | 27.03% | **87.50%** | 70.00% | 10.34% |
| **School** | Mousinho da Silveira (`MS`)* | 55 | 43.64% | **100.00%** | 66.67% | 20.51% |

*\*Note: Rural ($n=50$) and MS school ($n=55$) subgroups carry small-sample notices in the test set.*

---

## 🛠️ Tech Stack

- **Core ML & Data:** Python 3.11+, scikit-learn, XGBoost, pandas, NumPy
- **Explainability (XAI):** SHAP (TreeExplainer)
- **Generative AI:** Google Gemini API (`google-genai` SDK, `gemini-2.5-flash`) with Pydantic structured schemas
- **Database:** SQLAlchemy 2.x, SQLite (`eduguard.db`)
- **Dashboard & Visualization:** Streamlit, Plotly, Seaborn, Matplotlib
- **Testing & Quality:** Pytest (80 tests across 10 modules), python-dotenv

---

## 💻 How to Run Locally on Windows (PowerShell)

### 1. Clone Repository & Navigate
```powershell
git clone https://github.com/your-username/eduguard-ai.git
cd eduguard-ai
```

### 2. Set Up Virtual Environment
```powershell
# Create virtual environment
python -m venv .venv

# Activate environment in PowerShell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Configure Environment Variables (Optional for Gemini)
Create a `.env` file in the project root:
```ini
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
```
*(If no API key is provided, the platform automatically uses its built-in rule-based recommendation engine).*

### 5. Seed the SQLite Database
```powershell
python scripts/seed_db.py
```

### 6. Run the Unit Test Suite (80 Tests)
```powershell
pytest -v
```

### 7. Launch the Streamlit Multi-Page Web App
```powershell
streamlit run app.py
```

Open your browser at `http://localhost:8501` to view the dashboard:
- **`app.py`**: Executive summary, governance notices, and live telemetry.
- **`1_Admin_Dashboard.py`**: Institutional cohort KPIs, grade distributions, and alert feeds.
- **`2_Student_Analysis.py`**: Individual student lookups, TreeSHAP waterfall diagnosis, and analytical profile.
- **`3_What_If_Simulator.py`**: Interactive sensitivity simulator with non-causal guardrails.
- **`4_AI_Recommendations.py`**: 7-day personalized pedagogical plans via Gemini with deterministic fallback.
- **`5_Early_Warning.py`**: Longitudinal trend tracking, alert resolution queue, and intervention outcome tracker.
- **`6_Model_Explainability.py`**: Empirical regression/classification benchmarks, calibration curves, SHAP beeswarm/waterfalls, and fairness audits.

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
