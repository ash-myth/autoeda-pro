# ⌗ AutoEDA Pro

**Automated Exploratory Data Analysis & Machine Learning Platform**

Upload any CSV → get full statistical profiling, AutoML model comparison, feature importance, correlation analysis — in seconds.

---

## Features

- **Statistical Profiling** — mean, std, median, Q1/Q3, skewness, outlier detection (IQR) per column
- **Data Quality Scoring** — 0-100 quality score with issue detection (missing values, constant cols, high cardinality, skew)
- **Visualizations** — histograms, box plots, bar charts, correlation heatmap — all rendered with matplotlib/seaborn
- **Correlation Analysis** — Pearson correlation matrix with heatmap, ranked pair table
- **AutoML** — trains 5 models (Linear Regression, Ridge, Decision Tree, Random Forest, Gradient Boosting) with train/test split, cross-validation, R²/RMSE for regression or Accuracy/F1 for classification
- **Feature Importance** — Random Forest importances + correlation-based ranking
- **Residuals Analysis** — residual vs predicted plot + residual distribution
- **AI Insights(Coming soon)** — Gemini API generates professional analyst narrative conditioned on all EDA + ML results
- **PDF Export** — full multi-page report with all charts and stats tables

---

## Project Structure

```
autoeda_pro/
├── app.py           # Streamlit UI — all tabs, layout, state management
├── analysis.py      # EDA engine — per-column stats, quality scoring, issue detection
├── automl.py        # AutoML engine — scikit-learn model training, evaluation, feature importance
├── insights.py      # Gemini API integration — prompt engineering, error handling
├── report.py        # PDF report generation — matplotlib PdfPages, multi-page layout
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
