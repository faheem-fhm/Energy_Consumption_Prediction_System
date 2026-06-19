"""
Energy Consumption Prediction — Streamlit Application
Predicts household Global Active Power (kW) using XGBoost and Linear Regression.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import warnings
import io

warnings.filterwarnings("ignore")

# ─── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Energy Consumption Prediction",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Root variables */
    :root {
        --primary:    #0D47A1;
        --accent:     #1565C0;
        --highlight:  #42A5F5;
        --surface:    #F5F7FA;
        --border:     #D0D7E3;
        --text-main:  #1A1F2E;
        --text-muted: #5A6478;
        --success:    #2E7D32;
        --warning:    #E65100;
    }

    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', sans-serif;
        color: var(--text-main);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #0D1B2E;
    }
    [data-testid="stSidebar"] * {
        color: #CBD5E1 !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #FFFFFF !important;
        font-weight: 700;
    }

    /* Page header */
    .page-header {
        background: linear-gradient(135deg, #0D47A1 0%, #1976D2 60%, #42A5F5 100%);
        border-radius: 12px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        color: #fff;
    }
    .page-header h1 {
        margin: 0 0 0.3rem 0;
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #fff !important;
    }
    .page-header p {
        margin: 0;
        font-size: 0.95rem;
        opacity: 0.85;
        color: #fff !important;
    }

    /* Section card */
    .section-card {
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1.5rem 1.75rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    .section-card h3 {
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--primary);
        margin: 0 0 1rem 0;
        border-bottom: 2px solid var(--highlight);
        padding-bottom: 0.4rem;
    }

    /* KPI metric tiles */
    .kpi-row { display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1.25rem; }
    .kpi-tile {
        flex: 1;
        min-width: 140px;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1rem 1.25rem;
        text-align: center;
    }
    .kpi-tile .kpi-label {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--text-muted);
        margin-bottom: 0.35rem;
    }
    .kpi-tile .kpi-value {
        font-size: 1.55rem;
        font-weight: 800;
        color: var(--primary);
        line-height: 1.1;
    }
    .kpi-tile .kpi-sub {
        font-size: 0.72rem;
        color: var(--text-muted);
        margin-top: 0.2rem;
    }

    /* Prediction result box */
    .pred-box {
        background: linear-gradient(135deg, #E3F2FD, #BBDEFB);
        border: 2px solid #1565C0;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        text-align: center;
        margin-top: 1rem;
    }
    .pred-box .pred-label {
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #1565C0;
        margin-bottom: 0.4rem;
    }
    .pred-box .pred-value {
        font-size: 3rem;
        font-weight: 900;
        color: #0D47A1;
        line-height: 1;
    }
    .pred-box .pred-unit {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1565C0;
        margin-top: 0.2rem;
    }

    /* Status badges */
    .badge-success {
        display: inline-block;
        background: #E8F5E9;
        color: var(--success);
        border: 1px solid #A5D6A7;
        border-radius: 6px;
        padding: 0.15rem 0.6rem;
        font-size: 0.78rem;
        font-weight: 700;
    }

    /* Divider */
    .divider { border: none; border-top: 1px solid var(--border); margin: 1rem 0; }

    /* Streamlit widget label overrides */
    label { font-weight: 600 !important; font-size: 0.85rem !important; }

    /* Slider color */
    .stSlider > div > div > div > div { background: var(--primary) !important; }
</style>
""", unsafe_allow_html=True)


# ─── Session State ─────────────────────────────────────────────────────────────
if "models_trained" not in st.session_state:
    st.session_state.models_trained = False
if "df" not in st.session_state:
    st.session_state.df = None


# ─── Helper: train models ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_models(df_hash: int, _df: pd.DataFrame):
    """Train Linear Regression and XGBoost on the uploaded dataset."""
    num_cols = [
        "Global_active_power", "Global_reactive_power", "Voltage",
        "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
    ]
    for col in num_cols:
        _df[col] = pd.to_numeric(_df[col], errors="coerce")
        _df[col].fillna(_df[col].median(), inplace=True)

    _df["Total_Sub_Metering"] = (
        _df["Sub_metering_1"] + _df["Sub_metering_2"] + _df["Sub_metering_3"]
    )

    feature_cols = [
        "Global_reactive_power", "Voltage", "Global_intensity",
        "Sub_metering_1", "Sub_metering_2", "Sub_metering_3", "Total_Sub_Metering",
    ]
    X = _df[feature_cols]
    y = _df["Global_active_power"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    xgb = XGBRegressor(
        n_estimators=100, learning_rate=0.1, max_depth=6,
        random_state=42, verbosity=0,
    )
    xgb.fit(X_train, y_train)

    results = {}
    for name, mdl in [("Linear Regression", lr), ("XGBoost", xgb)]:
        y_pred = mdl.predict(X_test)
        results[name] = {
            "mse":   float(mean_squared_error(y_test, y_pred)),
            "rmse":  float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "r2":    float(r2_score(y_test, y_pred)),
            "train_r2": float(mdl.score(X_train, y_train)),
        }

    return lr, xgb, results, _df, feature_cols


# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ Energy Predictor")
    st.markdown("---")

    st.markdown("### 1 · Upload Dataset")
    uploaded_file = st.file_uploader(
        "CSV file (Household Electricity Dataset)",
        type=["csv"],
        help="Upload the household electricity consumption CSV.",
    )

    st.markdown("---")
    st.markdown("### 2 · Select Model")
    model_choice = st.selectbox(
        "Prediction model",
        ["XGBoost", "Linear Regression"],
        index=0,
    )

    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        """
        Predicts **Global Active Power** (kW) from household
        sub-metering and electrical measurements.

        **Models:** Linear Regression · XGBoost
        """,
    )


# ─── Page Header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
  <h1>⚡ Household Energy Consumption Prediction</h1>
  <p>Upload your dataset, review model performance, and run real-time predictions.</p>
</div>
""", unsafe_allow_html=True)


# ─── No file state ─────────────────────────────────────────────────────────────
if uploaded_file is None:
    st.info(
        "👈 Upload the **Household Electricity Dataset CSV** in the sidebar to get started.",
        icon="📂",
    )
    st.stop()


# ─── Load & Train ──────────────────────────────────────────────────────────────
df_raw = pd.read_csv(uploaded_file)
df_hash = hash(df_raw.to_csv())  # cache key

with st.spinner("Training models — this may take a moment on large datasets…"):
    lr_model, xgb_model, metrics, df, feature_cols = train_models(df_hash, df_raw.copy())

active_model = xgb_model if model_choice == "XGBoost" else lr_model


# ═══════════════════════════════════════════════════════════════════════════════
# TAB LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════
tab_overview, tab_explore, tab_model, tab_predict = st.tabs([
    "📊  Dataset Overview",
    "🔍  Exploration",
    "🏆  Model Performance",
    "⚡  Predict",
])


# ───────────────────────────────────────────────────────────────────────────────
# TAB 1 · Dataset Overview
# ───────────────────────────────────────────────────────────────────────────────
with tab_overview:
    n_rows, n_cols = df.shape
    n_missing_before = df_raw.isnull().sum().sum()

    st.markdown('<div class="kpi-row">', unsafe_allow_html=True)
    kpi_html = f"""
    <div class="kpi-row">
      <div class="kpi-tile">
        <div class="kpi-label">Total Records</div>
        <div class="kpi-value">{n_rows:,}</div>
      </div>
      <div class="kpi-tile">
        <div class="kpi-label">Features</div>
        <div class="kpi-value">{len(feature_cols)}</div>
        <div class="kpi-sub">after engineering</div>
      </div>
      <div class="kpi-tile">
        <div class="kpi-label">Missing Values</div>
        <div class="kpi-value">{n_missing_before:,}</div>
        <div class="kpi-sub">imputed with median</div>
      </div>
      <div class="kpi-tile">
        <div class="kpi-label">Target</div>
        <div class="kpi-value">GAP</div>
        <div class="kpi-sub">Global Active Power (kW)</div>
      </div>
    </div>
    """
    st.markdown(kpi_html, unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown('<div class="section-card"><h3>First 5 Rows</h3>', unsafe_allow_html=True)
        st.dataframe(df_raw.head(), use_container_width=True, height=220)
        st.markdown('</div>', unsafe_allow_html=True)
    with col_r:
        st.markdown('<div class="section-card"><h3>Descriptive Statistics</h3>', unsafe_allow_html=True)
        num_df = df_raw.select_dtypes(include="number")
        st.dataframe(num_df.describe().round(4), use_container_width=True, height=220)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card"><h3>Column Data Types &amp; Missing Values</h3>', unsafe_allow_html=True)
    info_df = pd.DataFrame({
        "Column": df_raw.columns,
        "Non-Null Count": df_raw.notnull().sum().values,
        "Missing": df_raw.isnull().sum().values,
        "Dtype": [str(t) for t in df_raw.dtypes.values],
    })
    st.dataframe(info_df, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────────────────────
# TAB 2 · Exploration
# ───────────────────────────────────────────────────────────────────────────────
with tab_explore:
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-card"><h3>Correlation Heatmap</h3>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 5))
        num_cols_heat = [
            "Global_active_power", "Global_reactive_power", "Voltage",
            "Global_intensity", "Sub_metering_1", "Sub_metering_2",
            "Sub_metering_3", "Total_Sub_Metering",
        ]
        corr = df[num_cols_heat].corr()
        sns.heatmap(
            corr, annot=True, fmt=".2f", cmap="Blues",
            linewidths=0.5, ax=ax, annot_kws={"size": 7},
        )
        ax.tick_params(axis="x", labelsize=7, rotation=45)
        ax.tick_params(axis="y", labelsize=7, rotation=0)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="section-card"><h3>Global Active Power Distribution</h3>', unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        ax2.hist(df["Global_active_power"].dropna(), bins=60, color="#1565C0", edgecolor="white", linewidth=0.4)
        ax2.set_xlabel("Global Active Power (kW)", fontsize=9)
        ax2.set_ylabel("Frequency", fontsize=9)
        ax2.set_title("Target Variable Distribution", fontsize=10, fontweight="bold")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card"><h3>Electricity Usage Over Time (first 1 000 readings)</h3>', unsafe_allow_html=True)
    try:
        df["DateTime"] = pd.to_datetime(df["Date"] + " " + df["Time"], dayfirst=True, errors="coerce")
        plot_df = df.dropna(subset=["DateTime"]).sort_values("DateTime").head(1000)
        fig3, ax3 = plt.subplots(figsize=(12, 3.5))
        ax3.plot(plot_df["DateTime"], plot_df["Global_active_power"],
                 color="#1565C0", linewidth=0.8, alpha=0.85)
        ax3.set_xlabel("Timestamp", fontsize=9)
        ax3.set_ylabel("kW", fontsize=9)
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)
        fig3.autofmt_xdate()
        fig3.tight_layout()
        st.pyplot(fig3, use_container_width=True)
        plt.close(fig3)
    except Exception:
        st.warning("Could not parse Date/Time columns for time-series plot.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card"><h3>Feature Boxplots</h3>', unsafe_allow_html=True)
    box_cols = ["Global_reactive_power", "Voltage", "Global_intensity",
                "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
    fig4, axes = plt.subplots(1, len(box_cols), figsize=(14, 4))
    for ax_b, col in zip(axes, box_cols):
        ax_b.boxplot(df[col].dropna(), patch_artist=True,
                     boxprops=dict(facecolor="#BBDEFB", color="#1565C0"),
                     medianprops=dict(color="#E65100", linewidth=2),
                     whiskerprops=dict(color="#1565C0"),
                     capprops=dict(color="#1565C0"),
                     flierprops=dict(marker="o", markersize=2, color="#90CAF9"))
        ax_b.set_title(col.replace("_", "\n"), fontsize=7, fontweight="bold")
        ax_b.tick_params(labelsize=7)
        ax_b.spines["top"].set_visible(False)
        ax_b.spines["right"].set_visible(False)
    fig4.tight_layout()
    st.pyplot(fig4, use_container_width=True)
    plt.close(fig4)
    st.markdown('</div>', unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────────────────────
# TAB 3 · Model Performance
# ───────────────────────────────────────────────────────────────────────────────
with tab_model:
    st.markdown("#### Model Comparison")

    metric_rows = []
    for mname, mdata in metrics.items():
        metric_rows.append({
            "Model":       mname,
            "R² (Test)":   f"{mdata['r2']:.6f}",
            "R² (Train)":  f"{mdata['train_r2']:.6f}",
            "RMSE":        f"{mdata['rmse']:.6f}",
            "MSE":         f"{mdata['mse']:.6f}",
        })
    st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)

    col_l, col_r = st.columns(2)

    for i, (mname, mdata) in enumerate(metrics.items()):
        target_col = col_l if i == 0 else col_r
        with target_col:
            badge = "✅ Excellent" if mdata["r2"] > 0.99 else ("✔ Good" if mdata["r2"] > 0.85 else "⚠ Moderate")
            kpi_html = f"""
            <div class="section-card">
              <h3>{mname} &nbsp;<span class="badge-success">{badge}</span></h3>
              <div class="kpi-row">
                <div class="kpi-tile">
                  <div class="kpi-label">R² Test</div>
                  <div class="kpi-value">{mdata['r2']:.4f}</div>
                </div>
                <div class="kpi-tile">
                  <div class="kpi-label">R² Train</div>
                  <div class="kpi-value">{mdata['train_r2']:.4f}</div>
                </div>
                <div class="kpi-tile">
                  <div class="kpi-label">RMSE</div>
                  <div class="kpi-value">{mdata['rmse']:.4f}</div>
                  <div class="kpi-sub">kW</div>
                </div>
                <div class="kpi-tile">
                  <div class="kpi-label">MSE</div>
                  <div class="kpi-value">{mdata['mse']:.4f}</div>
                </div>
              </div>
            </div>
            """
            st.markdown(kpi_html, unsafe_allow_html=True)

    # Bar chart comparison
    st.markdown('<div class="section-card"><h3>R² Score — Side-by-Side Comparison</h3>', unsafe_allow_html=True)
    fig5, ax5 = plt.subplots(figsize=(6, 3))
    names = list(metrics.keys())
    r2_vals = [v["r2"] for v in metrics.values()]
    colors = ["#1565C0", "#42A5F5"]
    bars = ax5.bar(names, r2_vals, color=colors, width=0.4, edgecolor="white")
    ax5.set_ylim(0, 1.05)
    ax5.set_ylabel("R² Score", fontsize=9)
    for bar, val in zip(bars, r2_vals):
        ax5.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax5.spines["top"].set_visible(False)
    ax5.spines["right"].set_visible(False)
    fig5.tight_layout()
    st.pyplot(fig5, use_container_width=True)
    plt.close(fig5)
    st.markdown('</div>', unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────────────────────
# TAB 4 · Predict
# ───────────────────────────────────────────────────────────────────────────────
with tab_predict:
    st.markdown(f"#### Real-Time Prediction &nbsp; · &nbsp; Model: **{model_choice}**")
    st.markdown("Enter the electrical measurements below to estimate Global Active Power consumption.")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-card"><h3>Electrical Measurements</h3>', unsafe_allow_html=True)
        grp = st.number_input(
            "Global Reactive Power (kVAR)",
            min_value=0.0, max_value=2.0, value=0.136, step=0.001, format="%.3f",
        )
        voltage = st.number_input(
            "Voltage (V)",
            min_value=200.0, max_value=280.0, value=241.97, step=0.01, format="%.2f",
        )
        intensity = st.number_input(
            "Global Intensity (A)",
            min_value=0.0, max_value=50.0, value=10.6, step=0.1, format="%.1f",
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="section-card"><h3>Sub-Metering Readings</h3>', unsafe_allow_html=True)
        sm1 = st.number_input("Sub-Metering 1 — Kitchen (Wh)", min_value=0.0, max_value=100.0, value=0.0, step=0.5)
        sm2 = st.number_input("Sub-Metering 2 — Laundry (Wh)", min_value=0.0, max_value=100.0, value=1.0, step=0.5)
        sm3 = st.number_input("Sub-Metering 3 — HVAC / Water Heater (Wh)", min_value=0.0, max_value=100.0, value=17.0, step=0.5)
        total_sm = sm1 + sm2 + sm3
        st.info(f"**Total Sub-Metering:** {total_sm:.1f} Wh (auto-computed)")
        st.markdown('</div>', unsafe_allow_html=True)

    run_btn = st.button("⚡  Run Prediction", type="primary", use_container_width=True)

    if run_btn:
        sample = pd.DataFrame([{
            "Global_reactive_power": grp,
            "Voltage":               voltage,
            "Global_intensity":      intensity,
            "Sub_metering_1":        sm1,
            "Sub_metering_2":        sm2,
            "Sub_metering_3":        sm3,
            "Total_Sub_Metering":    total_sm,
        }])

        prediction = float(active_model.predict(sample)[0])

        st.markdown(f"""
        <div class="pred-box">
          <div class="pred-label">Predicted Global Active Power</div>
          <div class="pred-value">{prediction:.4f}</div>
          <div class="pred-unit">kilowatts (kW)</div>
        </div>
        """, unsafe_allow_html=True)

        # Context
        st.markdown("---")
        col_ctx1, col_ctx2, col_ctx3 = st.columns(3)
        col_ctx1.metric("Predicted Consumption", f"{prediction:.4f} kW")
        col_ctx2.metric("Estimated Hourly Load", f"{prediction * 1:.2f} kWh")
        col_ctx3.metric("Model Used", model_choice)

        # Input summary
        with st.expander("Input Summary", expanded=False):
            st.dataframe(sample.T.rename(columns={0: "Value"}), use_container_width=True)
