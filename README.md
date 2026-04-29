# ⌗ AutoEDA Pro
**Automated Exploratory Data Analysis & Machine Learning Platform**

Upload any CSV → get full statistical profiling, AutoML model comparison, feature importance, correlation analysis, and an AI analyst report — in seconds.

---

## Features

- **Statistical Profiling** — mean, std, median, Q1/Q3, skewness, outlier detection (IQR) per column
- **Data Quality Scoring** — 0–100 quality score with issue detection (missing values, constant cols, high cardinality, skew)
- **Visualizations** — histograms, box plots, bar charts, correlation heatmap rendered with matplotlib/seaborn
- **Correlation Analysis** — Pearson correlation matrix with heatmap + ranked pair table
- **AutoML** — trains 5 models (Logistic/Linear Regression, Ridge, Decision Tree, Random Forest, Gradient Boosting) with auto-detection of regression vs classification; evaluates on R²/RMSE or Accuracy/F1 accordingly
- **Feature Importance** — Random Forest impurity-based importance + correlation-based ranking
- **Residuals Analysis** — residual vs predicted plot + residual distribution (regression tasks)
- **AI Insights** — Llama 3.3-70b via Groq generates a four-part analyst report: domain inference, cross-feature reasoning, leakage detection, and a column-level action plan — all conditioned on actual EDA + ML results
- **Ask the Analyst** — chat interface for follow-up questions grounded in your dataset context
- **PDF Export** — full multi-page report with all charts and stats tables

---

## Project Structure

```
autoeda_pro/
├── app.py          # Streamlit UI — tabs, layout, state management
├── analysis.py     # EDA engine — per-column stats, quality scoring, issue detection
├── automl.py       # AutoML engine — sklearn model training, evaluation, feature importance
├── insights.py     # Groq/Llama integration — prompt engineering, leakage detection, analyst chat
└── requirements.txt
```

---

## Local Setup

```bash
git clone https://github.com/ash-myth/autoeda-pro
cd autoeda-pro
pip install -r requirements.txt
streamlit run app.py
```

Add your Groq API key when prompted in the AI Insights tab.  
Get one free at [console.groq.com](https://console.groq.com)

---

## Sample Output — Titanic Dataset

| Model | Accuracy | F1 |
|---|---|---|
| Random Forest | 0.7374 | 0.7176 |
| Logistic Regression | 0.7318 | 0.7180 |
| Gradient Boosting | 0.7207 | 0.7045 |
| Decision Tree | 0.7151 | 0.6838 |

Quality Score: **71/100** — flagged Age missingness (19.9%), Fare outliers (116), and high cardinality in Name/Ticket.
