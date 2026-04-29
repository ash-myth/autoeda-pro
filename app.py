import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from io import BytesIO
from analysis import run_eda
from automl import run_automl
from report import generate_pdf_report
from insights import generate_report, ask_analyst, stream_groq
st.set_page_config(
    page_title="AutoEDA Pro",
    page_icon="⌗",
    layout="wide",
    initial_sidebar_state="collapsed"
)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=DM+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background-color: #07080f; color: #dde1f0; }
.block-container { padding: 2rem 3rem; max-width: 1200px; }

h1, h2, h3 { font-family: 'IBM Plex Mono', monospace !important; }
.metric-card { background: #13141f; border: 1px solid #1e2035; border-radius: 12px; padding: 18px 20px; }
.section-label {
font-family: 'IBM Plex Mono', monospace;
font-size: 13px;
letter-spacing: 1px;
color: #9aa3c7;
font-weight: 600;
margin-bottom: 10px;
}
.badge { font-family: 'IBM Plex Mono', monospace; font-size: 10px; font-weight: 700; border-radius: 5px; padding: 2px 8px; display: inline-block; }
.badge-accent { background: rgba(79,142,255,0.12); color: #4f8eff; }
.badge-green { background: rgba(34,211,160,0.1); color: #22d3a0; }
.badge-red { background: rgba(255,85,114,0.1); color: #ff5572; }
.badge-yellow { background: rgba(245,197,66,0.1); color: #f5c542; }
div[data-testid="stTab"] button { font-family: 'IBM Plex Mono', monospace !important; font-size: 13px !important; }
div[data-testid="metric-container"] { background: #13141f; border: 1px solid #1e2035; border-radius: 12px; padding: 16px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="display:flex; align-items:center; gap:14px; padding:0 0 28px 0; border-bottom:1px solid #1e2035; margin-bottom:32px;">
  <span style="font-family:'IBM Plex Mono',monospace; font-size:26px; font-weight:800; color:#fff;">
    <span style="color:#4f8eff;">auto</span>eda <span style="font-size:12px; color:#5a5f7a; font-weight:400;">PRO</span>
  </span>
  <span style="font-family:'IBM Plex Mono',monospace; font-size:12px; color:#5a5f7a; margin-left:auto;">
    Automated EDA · AutoML · AI Insights
  </span>
</div>
""", unsafe_allow_html=True)
uploaded = None
df = None
st.markdown("#### Select Data Source")

data_source = st.radio(
    "Choose input method",
    ["Upload CSV", "Use Sample Dataset"]
)
if data_source == "Upload CSV":

    uploaded = st.file_uploader(
        "Upload a CSV dataset",
        type=["csv"]
    )

    if uploaded:
        df = pd.read_csv(uploaded)
else:
    sample = st.selectbox(
        "Choose Sample Dataset",
        ["None", "Titanic", "House Prices", "Heart Disease"]
    )

    if sample == "Titanic":
        df = pd.read_csv("samples/Titanic-Dataset.csv")
        uploaded = type("obj", (object,), {"name": "Titanic-Dataset.csv"})()

    elif sample == "House Prices":
        df = pd.read_csv("samples/Housing.csv")
        uploaded = type("obj", (object,), {"name": "Housing.csv"})()

    elif sample == "Heart Disease":
        df = pd.read_csv("samples/heart_dataset.csv")
        uploaded = type("obj", (object,), {"name": "heart_dataset.csv"})()

if uploaded is None:
    st.markdown("""
    <div style="background:#0e0f1a; border:2px dashed #1e2035; border-radius:16px; padding:60px 40px; text-align:center; margin-top:24px;">
      <div style="font-size:48px; margin-bottom:16px;">⌗</div>
      <div style="font-family:'IBM Plex Mono',monospace; font-size:20px; font-weight:700; color:#fff; margin-bottom:10px;">drop your csv above</div>
      <div style="font-size:14px; color:#5a5f7a; margin-bottom:28px;">any dataset — sales, health, finance, environmental, sports…</div>
      <div style="display:flex; flex-wrap:wrap; gap:8px; justify-content:center;">
        %s
      </div>
    </div>
    """ % "".join(f'<span style="font-family:IBM Plex Mono,monospace;font-size:11px;color:#4f8eff;background:rgba(79,142,255,0.1);border-radius:6px;padding:4px 12px;">{f}</span>'
                  for f in ["Statistical Profiling", "Outlier Detection", "Correlation Matrix", "AutoML + Feature Importance", "AI Narrative", "PDF Report"]),
    unsafe_allow_html=True)
    st.stop()

@st.cache_data(show_spinner=False)
def load_and_analyze(file_bytes, file_name):
    df = pd.read_csv(BytesIO(file_bytes))
    eda = run_eda(df)
    ml = run_automl(df)
    return df, eda, ml

with st.spinner("Analyzing dataset…"):
    if data_source == "Upload CSV":
        file_bytes = uploaded.getvalue()
        df, eda, ml_result = load_and_analyze(file_bytes, uploaded.name)

    else:
        file_bytes = df.to_csv(index=False).encode()
        df, eda, ml_result = load_and_analyze(file_bytes, uploaded.name)

c1, c2, c3, c4, c5, c6 = st.columns(6)
metrics = [
    ("Rows", f"{len(df):,}"),
    ("Columns", len(df.columns)),
    ("Numeric", eda["n_numeric"]),
    ("Categorical", eda["n_categorical"]),
    ("Quality", f"{eda['quality_score']}/100"),
    ("Issues", len(eda["issues"])),
]
for col, (label, val) in zip([c1,c2,c3,c4,c5,c6], metrics):
    col.metric(label, val)

st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

tabs = st.tabs([
    "Dataset Overview",
    "Column Analysis",
    "Data Quality",
    "Feature Relationships",
    "Model Insights",
    "AI Insights"
])

with tabs[0]:
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="section-label">Missing Values by Column</div>', unsafe_allow_html=True)
        missing = eda["missing_pct"].sort_values(ascending=False)
        missing = missing[missing > 0]
        if missing.empty:
            st.success("✓ No missing values detected")
        else:
            fig, ax = plt.subplots(figsize=(6, max(2, len(missing) * 0.4)))
            fig.patch.set_facecolor("#0e0f1a")
            ax.set_facecolor("#0e0f1a")
            colors = ["#ff5572" if v > 20 else "#f5c542" for v in missing.values]
            bars = ax.barh(missing.index, missing.values, color=colors, height=0.6)
            ax.set_xlabel("Missing %", color="#5a5f7a", fontsize=10)
            ax.tick_params(colors="#5a5f7a", labelsize=9)
            ax.spines[:].set_color("#1e2035")
            ax.xaxis.label.set_color("#5a5f7a")
            for spine in ax.spines.values():
                spine.set_color("#1e2035")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with col_right:
        st.markdown('<div class="section-label">Schema Overview</div>', unsafe_allow_html=True)
        schema_data = []
        for col in df.columns:
            info = eda["column_info"][col]
            schema_data.append({
                "Column": col,
                "Type": info["type"],
                "Unique": info["unique"],
                "Missing%": f"{info['missing_pct']}%",
                "Flag": "⚠ ID" if info.get("is_id") else ("⚠ Const" if info.get("is_constant") else "✓")
            })
        st.dataframe(pd.DataFrame(schema_data), width="stretch", hide_index=True)

    st.markdown('<div class="section-label" style="margin-top:24px">Detected Issues</div>', unsafe_allow_html=True)
    if not eda["issues"]:
        st.success("No issues detected")
    else:
        for issue in eda["issues"]:
            if issue["sev"] == "high":
                st.error(f"🔴 {issue['msg']}")
            elif issue["sev"] == "med":
                st.warning(f"🟡 {issue['msg']}")
            else:
                st.info(f"🔵 {issue['msg']}")

with tabs[1]:

    num_cols = [c for c in df.columns if eda["column_info"][c]["type"] == "numeric"]
    cat_cols = [c for c in df.columns if eda["column_info"][c]["type"] == "categorical"]

    if num_cols:

        st.markdown('<div class="section-label">Numeric Columns</div>', unsafe_allow_html=True)

        selected_col = st.selectbox(
            "Select Numeric Column",
            num_cols
        )

        series = df[selected_col].dropna()

        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(6,4))
            fig.patch.set_facecolor("#07080f")
            ax.set_facecolor("#0e0f1a")

            ax.hist(series, bins=25, color="#4f8eff", alpha=0.85)

            ax.axvline(
                series.mean(),
                color="#22d3a0",
                linestyle="--",
                linewidth=2,
                label=f"Mean: {series.mean():.2f}"
            )

            ax.axvline(
                series.median(),
                color="#f5c542",
                linestyle="--",
                linewidth=2,
                label=f"Median: {series.median():.2f}"
            )

            ax.set_title(
                f"{selected_col} Distribution",
                color="#dde1f0",
                fontsize=12
            )

            ax.tick_params(colors="#9aa3c7")

            for spine in ax.spines.values():
                spine.set_color("#1e2035")

            ax.legend()

            st.pyplot(fig)
            plt.close()

        with col2:

            fig, ax = plt.subplots(figsize=(6,2.5))
            fig.patch.set_facecolor("#07080f")
            ax.set_facecolor("#0e0f1a")

            ax.boxplot(
                series,
                vert=False,
                patch_artist=True,
                boxprops=dict(
                    facecolor=(79/255,142/255,255/255,0.25),
                    color="#4f8eff"
                ),
                medianprops=dict(
                    color="#22d3a0",
                    linewidth=2
                )
            )

            info = eda["column_info"][selected_col]

            ax.set_title(
                f"Boxplot — Skew: {info.get('skew','?')}  |  Outliers: {info.get('outliers','?')}",
                color="#dde1f0",
                fontsize=10
            )

            ax.tick_params(colors="#9aa3c7")

            for spine in ax.spines.values():
                spine.set_color("#1e2035")

            st.pyplot(fig)
            plt.close()


    if cat_cols:

        st.markdown(
            '<div class="section-label" style="margin-top:24px">Categorical Columns</div>',
            unsafe_allow_html=True
        )

        selected_cat = st.selectbox(
            "Select Categorical Column",
            cat_cols
        )

        vc = df[selected_cat].value_counts().head(10)

        fig, ax = plt.subplots(figsize=(7,4))
        fig.patch.set_facecolor("#07080f")
        ax.set_facecolor("#0e0f1a")

        ax.barh(
            vc.index.astype(str),
            vc.values,
            color="#b87fff",
            alpha=0.85
        )

        ax.set_title(
            f"{selected_cat} Top Categories",
            color="#dde1f0",
            fontsize=12
        )

        ax.tick_params(colors="#9aa3c7")

        for spine in ax.spines.values():
            spine.set_color("#1e2035")

        ax.invert_yaxis()

        st.pyplot(fig)
        plt.close()

with tabs[2]:
    col_gauge, col_issues = st.columns([1, 2])

    with col_gauge:
        st.markdown('<div class="section-label">Quality Score</div>', unsafe_allow_html=True)
        score = eda["quality_score"]
        color = "#22d3a0" if score > 75 else "#f5c542" if score > 50 else "#ff5572"
        fig, ax = plt.subplots(figsize=(3.5, 3.5), subplot_kw=dict(polar=True))
        fig.patch.set_facecolor("#0e0f1a")
        ax.set_facecolor("#0e0f1a")
        theta = np.linspace(0, np.pi, 100)
        ax.plot(theta, [1]*100, color="#1e2035", linewidth=12)
        fill_theta = np.linspace(0, np.pi * score / 100, 100)
        ax.plot(fill_theta, [1]*100, color=color, linewidth=12)
        ax.set_ylim(0, 1.5)
        ax.set_xlim(0, np.pi)
        ax.axis("off")
        ax.text(np.pi/2, 0.3, str(score), ha="center", va="center", fontsize=36, fontweight="bold", color="#ffffff", fontfamily="monospace")
        ax.text(np.pi/2, 0.05, "/100", ha="center", va="center", fontsize=12, color="#5a5f7a", fontfamily="monospace")
        label = "GOOD" if score > 75 else "MODERATE" if score > 50 else "NEEDS WORK"
        ax.text(np.pi/2, -0.2, label, ha="center", va="center", fontsize=10, color=color, fontfamily="monospace", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_issues:
        st.markdown('<div class="section-label">Issues Breakdown</div>', unsafe_allow_html=True)
        if not eda["issues"]:
            st.success("✓ No issues detected")
        else:
            for issue in eda["issues"]:
                if issue["sev"] == "high":
                    st.error(f"🔴 **[HIGH]** {issue['msg']}")
                elif issue["sev"] == "med":
                    st.warning(f"🟡 **[MED]** {issue['msg']}")
                else:
                    st.info(f"🔵 **[INFO]** {issue['msg']}")

    st.markdown('<div class="section-label" style="margin-top:24px">Column Health Report</div>', unsafe_allow_html=True)
    health_rows = []
    for col in df.columns:
        info = eda["column_info"][col]
        bad = info["missing_pct"] > 20 or info.get("is_constant") or info.get("is_id")
        health_rows.append({
        "Column": col,
        "Type": info["type"],
        "Missing%": info["missing_pct"],
        "Unique": info["unique"],
        "Mean/Top": info.get("mean", info.get("top")),
        "Std/Count": info.get("std", info.get("count")),
        "Skew": info.get("skew", None),
        "Outliers": info.get("outliers", None),
    })
    st.dataframe(pd.DataFrame(health_rows), width="stretch", hide_index=True)

with tabs[3]:

    st.markdown('<div class="section-label">Correlation Analysis</div>', unsafe_allow_html=True)

    numeric_df = df.select_dtypes(include=np.number)

    if len(numeric_df.columns) > 1:

        corr = numeric_df.corr()

        plot_type = st.selectbox(
            "Select Analysis Type",
            [
                "Heatmap",
                "Scatter Plot",
                "Top Correlations",
                "Correlation Table"
            ]
        )

        if plot_type == "Heatmap":

            fig, ax = plt.subplots(figsize=(10,7))
            fig.patch.set_facecolor("#07080f")
            ax.set_facecolor("#0e0f1a")

            sns.heatmap(
                corr,
                cmap="coolwarm",
                center=0,
                annot=False,
                ax=ax
            )

            ax.set_title(
                "Correlation Heatmap",
                color="#dde1f0"
            )

            ax.tick_params(colors="#9aa3c7")

            st.pyplot(fig)
            plt.close()


        elif plot_type == "Scatter Plot":

            col1, col2 = st.columns(2)

            with col1:
                feature_x = st.selectbox(
                    "Feature 1",
                    numeric_df.columns
                )

            with col2:
                feature_y = st.selectbox(
                    "Feature 2",
                    numeric_df.columns,
                    index=1 if len(numeric_df.columns) > 1 else 0
                )

            fig, ax = plt.subplots(figsize=(7,4))
            fig.patch.set_facecolor("#07080f")
            ax.set_facecolor("#0e0f1a")

            ax.scatter(
                df[feature_x],
                df[feature_y],
                alpha=0.6,
                color="#4f8eff"
            )

            corr_val = df[feature_x].corr(df[feature_y])

            ax.set_title(
                f"{feature_x} vs {feature_y} (corr={corr_val:.2f})",
                color="#dde1f0"
            )

            ax.tick_params(colors="#9aa3c7")

            st.pyplot(fig)
            plt.close()


        elif plot_type == "Top Correlations":

            corr_pairs = (
                corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                .stack()
                .reset_index()
            )

            corr_pairs.columns = ["Feature 1", "Feature 2", "Correlation"]

            top_corr = corr_pairs.sort_values(
                "Correlation",
                key=abs,
                ascending=False
            ).head(15)

            st.dataframe(
                top_corr,
                width="stretch"
            )


        elif plot_type == "Correlation Table":

            st.dataframe(
                corr,
                width="stretch"
            )

    else:
        st.info("Not enough numeric columns for correlation analysis.")
with tabs[4]:
    if ml_result is None:
        st.warning("Need at least 2 numeric columns to run AutoML.")
    else:
        st.markdown(f"""
        <div style="background:#13141f; border:1px solid #1e2035; border-radius:12px; padding:18px 22px; margin-bottom:20px;">
          <div class="section-label">Experiment Config</div>
          <div style="display:grid; grid-template-columns:repeat(5,1fr); gap:16px; margin-top:12px;">
            {''.join(f'<div><div style="font-family:IBM Plex Mono,monospace;font-size:9px;color:#5a5f7a;letter-spacing:1px;">{k}</div><div style="font-family:IBM Plex Mono,monospace;font-size:15px;font-weight:700;color:#fff;margin-top:4px;">{v}</div></div>'
                     for k,v in [("TASK", ml_result["task"]), ("TARGET", ml_result["target"]),
                                  ("FEATURES", len(ml_result["features"])),
                                  ("TRAIN", ml_result["train_size"]), ("TEST", ml_result["test_size"])])}
          </div>
        </div>
        """, unsafe_allow_html=True)

        col_models, col_imp = st.columns(2)

        with col_models:
            st.markdown('<div class="section-label">Model Comparison</div>', unsafe_allow_html=True)
            for m in ml_result["models"]:
                is_best = m["name"] == ml_result["best_model"]
                border_color = "#4f8eff" if is_best else "#1e2035"
                metric_key = "Accuracy" if ml_result["task"] == "classification" else "R²"
                metric_val = m.get("accuracy", m.get("r2", "—"))
                st.markdown(f"""
                <div style="background:{'rgba(79,142,255,0.08)' if is_best else '#0e0f1a'}; border:1px solid {border_color}; border-radius:12px; padding:16px 18px; margin-bottom:12px;">
                  <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                    <span style="font-family:IBM Plex Mono,monospace; font-size:13px; font-weight:700; color:#fff;">{m['name']}</span>
                    {'<span class="badge badge-green">BEST</span>' if is_best else ''}
                  </div>
                  <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px;">
                    <div style="background:#13141f; border-radius:8px; padding:10px 14px;">
                      <div style="font-family:IBM Plex Mono,monospace; font-size:9px; color:#5a5f7a; margin-bottom:4px;">{metric_key}</div>
                      <div style="font-family:IBM Plex Mono,monospace; font-size:22px; font-weight:800; color:{'#22d3a0' if float(str(metric_val)) > 0.7 else '#f5c542' if float(str(metric_val)) > 0.4 else '#ff5572'};">{metric_val}</div>
                    </div>
                    <div style="background:#13141f; border-radius:8px; padding:10px 14px;">
                      <div style="font-family:IBM Plex Mono,monospace; font-size:9px; color:#5a5f7a; margin-bottom:4px;">{'F1' if ml_result['task'] == 'classification' else 'RMSE'}</div>
                      <div style="font-family:IBM Plex Mono,monospace; font-size:22px; font-weight:800; color:#dde1f0;">{m.get('f1', m.get('rmse', '—'))}</div>
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

        with col_imp:
            st.markdown('<div class="section-label">Feature Importance</div>', unsafe_allow_html=True)
            imp = ml_result["feature_importance"]
            if imp:
                fig, ax = plt.subplots(figsize=(5, max(3, len(imp) * 0.5)))
                fig.patch.set_facecolor("#0e0f1a")
                ax.set_facecolor("#0e0f1a")
                names = [x["col"] for x in imp]
                vals = [x["importance"] for x in imp]
                colors_bar = ["#4f8eff" if i == 0 else "#2d3a5e" for i in range(len(names))]
                ax.barh(names, vals, color=colors_bar, height=0.6)
                ax.invert_yaxis()
                ax.set_xlabel("Importance", color="#5a5f7a", fontsize=9)
                ax.tick_params(colors="#5a5f7a", labelsize=9)
                ax.spines[:].set_color("#1e2035")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

        if "confusion_matrix" in ml_result:
            st.markdown('<div class="section-label" style="margin-top:16px">Confusion Matrix</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 4))
            fig.patch.set_facecolor("#0e0f1a")
            ax.set_facecolor("#0e0f1a")
            cm = np.array(ml_result["confusion_matrix"])
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                        linewidths=0.5, linecolor="#1e2035",
                        annot_kws={"color": "white", "size": 11})
            ax.tick_params(colors="#dde1f0", labelsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        if "residuals" in ml_result:
            st.markdown('<div class="section-label" style="margin-top:16px">Residuals Plot</div>', unsafe_allow_html=True)
            fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
            fig.patch.set_facecolor("#0e0f1a")
            for ax in axes:
                ax.set_facecolor("#0e0f1a")
                ax.tick_params(colors="#5a5f7a", labelsize=8)
                ax.spines[:].set_color("#1e2035")

            preds = np.array(ml_result["predictions"])
            actuals = np.array(ml_result["actuals"])
            residuals = actuals - preds

            axes[0].scatter(preds, residuals, color="#4f8eff", alpha=0.5, s=20)
            axes[0].axhline(0, color="#22d3a0", linewidth=1.5, linestyle="--")
            axes[0].set_xlabel("Predicted", color="#5a5f7a", fontsize=9)
            axes[0].set_ylabel("Residuals", color="#5a5f7a", fontsize=9)
            axes[0].set_title("Residuals vs Predicted", color="#dde1f0", fontsize=10)

            axes[1].hist(residuals, bins=25, color="#b87fff", alpha=0.8, edgecolor="#07080f")
            axes[1].set_xlabel("Residual", color="#5a5f7a", fontsize=9)
            axes[1].set_title("Residual Distribution", color="#dde1f0", fontsize=10)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

with tabs[5]:

    groq_key = None
    try:
        groq_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

    if not groq_key:
        groq_key = st.session_state.get("groq_api_key", "")

    if not groq_key:
        st.markdown("""
        <div style="background:#13141f; border:1px solid #1e2035; border-radius:12px;
                    padding:28px 28px; margin-bottom:24px;">
          <div style="font-family:'IBM Plex Mono',monospace; font-size:13px;
                      font-weight:700; color:#fff; margin-bottom:6px;">✦ AI Insights</div>
          <div style="font-size:13px; color:#9aa3c7; margin-bottom:18px;">
            Paste your Groq API key to activate the analyst. Keys are free at
            <a href="https://console.groq.com" target="_blank"
               style="color:#4f8eff;">console.groq.com</a>
            and never stored beyond this session.
          </div>
        </div>
        """, unsafe_allow_html=True)

        key_input = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_…",
        )
        if key_input:
            st.session_state["groq_api_key"] = key_input
            st.rerun()
        st.stop()

    st.markdown("""
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:24px;">
      <span style="font-family:'IBM Plex Mono',monospace; font-size:18px;
                   font-weight:800; color:#fff;">✦ AI Analyst Report</span>
      <span class="badge badge-accent">llama-3.3-70b · groq</span>
    </div>
    """, unsafe_allow_html=True)

    report_cache_key = f"ai_report_{uploaded.name}_{eda['quality_score']}"

    if report_cache_key not in st.session_state:
        with st.spinner("Analyzing dataset with AI…"):
            try:
                st.session_state[report_cache_key] = generate_report(
                    groq_key, df, eda, ml_result
                )
            except Exception as e:
                st.error(f"Groq API error: {e}")
                st.stop()

    report = st.session_state[report_cache_key]

    col_a, col_b = st.columns([1, 1])

    col_a.markdown(f"""
    <div style="background:#13141f; border:1px solid #1e2035; border-left:3px solid #4f8eff;
                border-radius:12px; padding:20px 22px; margin-bottom:18px; height:100%;">
      <div style="font-family:'IBM Plex Mono',monospace; font-size:10px; letter-spacing:1px;
                  color:#4f8eff; font-weight:700; margin-bottom:10px;">DOMAIN INFERENCE</div>
      <div style="font-size:13.5px; color:#dde1f0; line-height:1.75;">{report.get("domain", "—")}</div>
    </div>
    """, unsafe_allow_html=True)

    col_b.markdown(f"""
    <div style="background:#13141f; border:1px solid #1e2035; border-left:3px solid #f5c542;
                border-radius:12px; padding:20px 22px; margin-bottom:18px; height:100%;">
      <div style="font-family:'IBM Plex Mono',monospace; font-size:10px; letter-spacing:1px;
                  color:#f5c542; font-weight:700; margin-bottom:10px;">CROSS-FEATURE REASONING</div>
      <div style="font-size:13.5px; color:#dde1f0; line-height:1.75;">{report.get("cross_feature", "—")}</div>
    </div>
    """, unsafe_allow_html=True)

    col_c, col_d = st.columns([1, 1])

    suspects = report.get("_leakage_suspects", [])
    if suspects:
        badge_html = " ".join(
            f'<span style="font-family:IBM Plex Mono,monospace; font-size:10px; font-weight:700; '
            f'border-radius:5px; padding:3px 9px; display:inline-block; margin:2px; '
            f'background:{"rgba(255,85,114,0.15)" if s["level"] == "critical" else "rgba(245,197,66,0.12)"}; '
            f'color:{"#ff5572" if s["level"] == "critical" else "#f5c542"};">'
            f'{s["col"]} r={s["r"]}</span>'
            for s in suspects
        )
        leakage_badge_block = f'<div style="margin-bottom:10px;">{badge_html}</div>'
    else:
        leakage_badge_block = ""

    leakage_border = "#ff5572" if suspects else "#22d3a0"
    col_c.markdown(f"""
    <div style="background:#13141f; border:1px solid #1e2035; border-left:3px solid {leakage_border};
                border-radius:12px; padding:20px 22px; margin-bottom:18px;">
      <div style="font-family:'IBM Plex Mono',monospace; font-size:10px; letter-spacing:1px;
                  color:{leakage_border}; font-weight:700; margin-bottom:10px;">LEAKAGE ANALYSIS</div>
      {leakage_badge_block}
      <div style="font-size:13.5px; color:#dde1f0; line-height:1.75;">{report.get("leakage", "—")}</div>
    </div>
    """, unsafe_allow_html=True)

    action_raw = report.get("action_plan", "—")
    # LLM may return a list of alternating [num, text, num, text] or plain strings
    if isinstance(action_raw, list):
        # Filter out bare numbers, keep only string items
        step_texts = [str(item) for item in action_raw if not isinstance(item, (int, float))]
        # If filtering left nothing, stringify everything non-numeric
        if not step_texts:
            step_texts = [str(item) for item in action_raw]
    elif isinstance(action_raw, str):
        step_texts = [l.strip() for l in action_raw.strip().splitlines() if l.strip()]
    else:
        step_texts = [str(action_raw)]

    steps_html = "".join(
        f'<div style="display:flex; gap:10px; margin-bottom:12px; align-items:flex-start;">'
        f'<span style="font-family:IBM Plex Mono,monospace; font-size:10px; font-weight:700; '
        f'color:#b87fff; background:rgba(184,127,255,0.1); border-radius:4px; '
        f'padding:2px 7px; white-space:nowrap; margin-top:2px; flex-shrink:0;">{i+1}</span>'
        f'<span style="font-size:13px; color:#dde1f0; line-height:1.65;">'
        f'{text.lstrip("0123456789. ")}</span>'
        f'</div>'
        for i, text in enumerate(step_texts)
    )

    col_d.markdown(f"""
    <div style="background:#13141f; border:1px solid #1e2035; border-left:3px solid #b87fff;
                border-radius:12px; padding:20px 22px; margin-bottom:18px;">
      <div style="font-family:'IBM Plex Mono',monospace; font-size:10px; letter-spacing:1px;
                  color:#b87fff; font-weight:700; margin-bottom:14px;">ACTION PLAN</div>
      {steps_html}
    </div>
    """, unsafe_allow_html=True)

    if st.button("↺ Regenerate Report", key="regen_report"):
        if report_cache_key in st.session_state:
            del st.session_state[report_cache_key]
        st.rerun()

    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace; font-size:13px; font-weight:700;
                color:#fff; letter-spacing:0.5px; margin-bottom:16px;">
      ✦ Ask the Analyst
    </div>
    """, unsafe_allow_html=True)

    chat_key = f"ai_chat_{uploaded.name}"
    if chat_key not in st.session_state:
        st.session_state[chat_key] = []

    # Render full history with native chat_message (no rerender flicker)
    for msg in st.session_state[chat_key]:
        with st.chat_message("user" if msg["role"] == "user" else "assistant"):
            st.markdown(msg["content"])

    question = st.chat_input("Ask anything about this dataset…")
    if question:
        # Append + immediately render user bubble
        st.session_state[chat_key].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        history_for_api = st.session_state[chat_key][-6:]
        system = (
            "You are a senior ML engineer. The user is asking follow-up questions about their dataset.\n"
            "Answer concisely and specifically — cite actual column names, correlation values, "
            "model scores, or feature importances from the context.\n"
            "Do NOT restate what the user can already see in charts. "
            "Reason about implications, causes, and what to do next.\n"
            "Keep answers under 120 words."
        )
        messages = [{"role": "system", "content": system}]
        messages.extend(history_for_api[:-1])
        messages.append({"role": "user", "content": question})

        try:
            with st.chat_message("assistant"):
                answer = st.write_stream(stream_groq(groq_key, messages))
            st.session_state[chat_key].append({"role": "assistant", "content": answer})
        except Exception as e:
            err = f"⚠ API error: {e}"
            st.session_state[chat_key].append({"role": "assistant", "content": err})
            with st.chat_message("assistant"):
                st.error(err)

    if st.session_state.get(chat_key):
        if st.button("✕ Clear chat", key="clear_chat"):
            st.session_state[chat_key] = []
            st.rerun()

with st.sidebar:
    st.markdown("### ⌗ AutoEDA Pro")
    st.markdown("---")
    groq_sidebar_key = None
    try:
        groq_sidebar_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    if not groq_sidebar_key:
        groq_sidebar_key = st.session_state.get("groq_api_key", "")

    if groq_sidebar_key:
        st.markdown(
            "<div style='font-family:IBM Plex Mono,monospace;font-size:11px;"
            "color:#22d3a0;margin-bottom:4px;'>✓ Groq API key active</div>",
            unsafe_allow_html=True
        )
        if st.button("✕ Clear key", key="clear_groq_key"):
            st.session_state.pop("groq_api_key", None)
            st.rerun()
    else:
        st.markdown(
            "<div style='font-family:IBM Plex Mono,monospace;font-size:11px;"
            "color:#5a5f7a;margin-bottom:4px;'>○ Groq key not set — see AI Insights tab</div>",
            unsafe_allow_html=True
        )

    st.markdown("---")

    st.markdown("**Export Report**")

    if st.button("📄 Click to Generate PDF Report"):
        with st.spinner("Generating PDF…"):
            pdf_bytes = generate_pdf_report(
                df,
                eda,
                ml_result,
                uploaded.name
            )

        st.download_button(
            label="⬇ Download",
            data=pdf_bytes,
            file_name=f"autoeda_{uploaded.name.replace('.csv','')}.pdf",
            mime="application/pdf"
        )

    st.markdown("---")

    st.markdown(f"**File:** {uploaded.name}")
    st.markdown(f"**Rows:** {len(df):,}")
    st.markdown(f"**Columns:** {len(df.columns)}")

    if ml_result:
        st.markdown(f"**Best Model:** {ml_result['best_model']}")
    
    st.markdown("""
    ---
    <div style='text-align:center;color:#5a5f7a;font-size:12px'>
    Built by Ashmit Chatterjee ·
    <a href='https://github.com/ash-myth/autoeda-pro' target='_blank' style='color:#4f8eff;text-decoration:none'>GitHub</a>
    </div>
    """, unsafe_allow_html=True)
