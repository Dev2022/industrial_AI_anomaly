# ============================================================
# ADVANCED STREAMLIT DASHBOARD
# dashboard/streamlit_app.py
# ============================================================

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AVATHON Predictive Maintenance",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

h1, h2, h3 {
    color: white;
}

.metric-card {
    background-color: #1E1E1E;
    padding: 15px;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================

st.title("🏭 AVATHON Industrial AI Platform")

st.markdown("""
End-to-end predictive maintenance and optimization platform.

### Features
- Industrial sensor simulation
- Anomaly detection
- Failure prediction
- RUL estimation
- SHAP explainability
- MILP optimization
- Scenario analysis
""")

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    return pd.read_csv(
        "results/final_dataset.csv"
    )

@st.cache_data
def load_maintenance():

    return pd.read_csv(
        "results/maintenance_plan.csv"
    )

@st.cache_data
def load_scenario():

    scenario_path = (
        "results/scenario_analysis.csv"
    )

    if os.path.exists(scenario_path):

        return pd.read_csv(scenario_path)

    return None

df = load_data()
maintenance_df = load_maintenance()
scenario_df = load_scenario()

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙ Dashboard Controls")

machine_list = sorted(
    df["machine_id"].unique()
)

selected_machine = st.sidebar.selectbox(
    "Select Machine",
    machine_list
)

sensor_choice = st.sidebar.selectbox(

    "Primary Sensor",

    [
        "temperature",
        "vibration",
        "acoustic_emission",
        "pressure",
        "rpm",
        "current"
    ]
)

window_size = st.sidebar.slider(
    "Rolling Window",
    10,
    120,
    60
)

show_anomalies = st.sidebar.checkbox(
    "Show Anomalies",
    value=True
)

# ============================================================
# MACHINE DATA
# ============================================================

machine_df = df[
    df["machine_id"] == selected_machine
].copy()

latest = machine_df.iloc[-1]

# ============================================================
# TOP KPIs
# ============================================================

st.header("📌 Live Machine KPIs")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "Temperature",
    f"{latest['temperature']:.2f} °C"
)

col2.metric(
    "Vibration",
    f"{latest['vibration']:.3f}"
)

col3.metric(
    "Pressure",
    f"{latest['pressure']:.2f}"
)

col4.metric(
    "Failure Risk",
    f"{latest['predicted_failure_probability']:.2%}"
)

rul_col = (
    "predicted_RUL"
    if "predicted_RUL" in machine_df.columns
    else None
)

if rul_col:

    col5.metric(
        "Predicted RUL",
        f"{latest[rul_col]:.1f}"
    )

# ============================================================
# HEALTH STATUS
# ============================================================

risk = latest["predicted_failure_probability"]

if risk < 0.30:

    st.success("✅ Machine Status: HEALTHY")

elif risk < 0.70:

    st.warning("⚠ Machine Status: MODERATE RISK")

else:

    st.error("🚨 Machine Status: HIGH RISK")

# ============================================================
# RAW SENSOR VISUALIZATION
# ============================================================

st.header("📈 Raw Sensor Trends")

fig1, ax1 = plt.subplots(
    figsize=(14, 5)
)

ax1.plot(
    machine_df["timestamp"],
    machine_df[sensor_choice],
    linewidth=2
)

ax1.set_xlabel("Timestamp")
ax1.set_ylabel(sensor_choice)

ax1.set_title(
    f"{sensor_choice} Trend"
)

st.pyplot(fig1)

# ============================================================
# MULTI SENSOR VIEW
# ============================================================

st.header("📊 Multi-Sensor Monitoring")

fig2, ax2 = plt.subplots(
    figsize=(14, 6)
)

ax2.plot(
    machine_df["timestamp"],
    machine_df["temperature"],
    label="Temperature"
)

ax2.plot(
    machine_df["timestamp"],
    machine_df["vibration"] * 100,
    label="Vibration x100"
)

ax2.plot(
    machine_df["timestamp"],
    machine_df["current"],
    label="Current"
)

ax2.plot(
    machine_df["timestamp"],
    machine_df["pressure"],
    label="Pressure"
)

ax2.legend()

st.pyplot(fig2)

# ============================================================
# WINDOW FEATURES
# ============================================================

st.header("🧠 Rolling Window Features")

rolling_mean = (
    machine_df[sensor_choice]
    .rolling(window_size)
    .mean()
)

rolling_std = (
    machine_df[sensor_choice]
    .rolling(window_size)
    .std()
)

fig3, ax3 = plt.subplots(
    figsize=(14, 5)
)

ax3.plot(
    machine_df["timestamp"],
    machine_df[sensor_choice],
    alpha=0.4,
    label="Raw Signal"
)

ax3.plot(
    machine_df["timestamp"],
    rolling_mean,
    linewidth=2,
    label="Rolling Mean"
)

ax3.plot(
    machine_df["timestamp"],
    rolling_std,
    linewidth=2,
    label="Rolling Std"
)

ax3.legend()

st.pyplot(fig3)

# ============================================================
# ANOMALY DETECTION
# ============================================================

st.header("🚨 Anomaly Detection")

if (
    show_anomalies
    and "anomaly_label" in machine_df.columns
):

    anomaly_df = machine_df[
        machine_df["anomaly_label"] == 1
    ]

    fig4, ax4 = plt.subplots(
        figsize=(14, 5)
    )

    ax4.plot(
        machine_df["timestamp"],
        machine_df[sensor_choice],
        label=sensor_choice
    )

    ax4.scatter(
        anomaly_df["timestamp"],
        anomaly_df[sensor_choice],
        s=60,
        label="Anomaly"
    )

    ax4.legend()

    st.pyplot(fig4)

    st.subheader("Latest Anomalies")

    display_cols = [

        "timestamp",
        "temperature",
        "vibration",
        "pressure",
        "rpm",
        "current",
        "predicted_failure_probability"
    ]

    valid_cols = [

        c for c in display_cols
        if c in anomaly_df.columns
    ]

    st.dataframe(
        anomaly_df[valid_cols].tail(20),
        use_container_width=True
    )

# ============================================================
# FAILURE PROBABILITY
# ============================================================

st.header("⚠ Failure Probability Trend")

fig5, ax5 = plt.subplots(
    figsize=(14, 5)
)

ax5.plot(
    machine_df["timestamp"],
    machine_df[
        "predicted_failure_probability"
    ],
    linewidth=2
)

ax5.set_ylim(0, 1)

ax5.set_ylabel(
    "Failure Probability"
)

st.pyplot(fig5)

# ============================================================
# RUL TREND
# ============================================================

if rul_col:

    st.header("⏳ Remaining Useful Life")

    fig6, ax6 = plt.subplots(
        figsize=(14, 5)
    )

    ax6.plot(
        machine_df["timestamp"],
        machine_df[rul_col],
        linewidth=2
    )

    ax6.set_ylabel("Predicted RUL")

    st.pyplot(fig6)

# ============================================================
# SHAP EXPLAINABILITY
# ============================================================

st.header("🧠 SHAP Explainability")

shap_path = "results/shap_summary.png"

if os.path.exists(shap_path):

    st.image(
        shap_path,
        use_container_width=True
    )

else:

    st.warning(
        "SHAP summary image not found."
    )

# ============================================================
# OPTIMIZATION RESULTS
# ============================================================

st.header("🛠 Maintenance Optimization")

st.dataframe(
    maintenance_df.sort_values(
        "failure_probability",
        ascending=False
    ),
    use_container_width=True
)

# ============================================================
# RISK DISTRIBUTION
# ============================================================

st.subheader("Machine Risk Ranking")

fig7, ax7 = plt.subplots(
    figsize=(12, 5)
)

sorted_df = maintenance_df.sort_values(
    "failure_probability",
    ascending=False
)

ax7.bar(
    sorted_df["machine_id"].astype(str),
    sorted_df["failure_probability"]
)

ax7.set_xlabel("Machine ID")

ax7.set_ylabel(
    "Failure Probability"
)

st.pyplot(fig7)

# ============================================================
# MAINTENANCE ACTIONS
# ============================================================

st.subheader("Maintenance Actions")

maintenance_actions = maintenance_df[
    maintenance_df[
        "maintenance_action"
    ] == 1
]

st.dataframe(
    maintenance_actions,
    use_container_width=True
)

# ============================================================
# SCENARIO ANALYSIS
# ============================================================

if scenario_df is not None:

    st.header("📊 Scenario Analysis")

    fig8, ax8 = plt.subplots(
        figsize=(12, 5)
    )

    ax8.plot(
        scenario_df["technician_hours"],
        scenario_df["optimized_cost"],
        marker="o"
    )

    ax8.set_xlabel(
        "Technician Hours"
    )

    ax8.set_ylabel(
        "Optimized Cost"
    )

    st.pyplot(fig8)

    st.dataframe(
        scenario_df,
        use_container_width=True
    )

# ============================================================
# SYSTEM SUMMARY
# ============================================================

st.header("📌 System Summary")

total_machines = (
    df["machine_id"].nunique()
)

total_anomalies = (

    int(df["anomaly_label"].sum())

    if "anomaly_label" in df.columns
    else 0
)

high_risk = maintenance_df[
    maintenance_df[
        "failure_probability"
    ] > 0.7
].shape[0]

normal = maintenance_df[
    maintenance_df[
        "failure_probability"
    ] < 0.3
].shape[0]

moderate = (
    total_machines
    - high_risk
    - normal
)

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Machines",
    total_machines
)

c2.metric(
    "Anomalies",
    total_anomalies
)

c3.metric(
    "High Risk",
    high_risk
)

c4.metric(
    "Moderate Risk",
    moderate
)

# ============================================================
# DOWNLOAD SECTION
# ============================================================

st.header("⬇ Download Results")

with open(
    "results/maintenance_plan.csv",
    "rb"
) as file:

    st.download_button(

        label="Download Maintenance Plan",

        data=file,

        file_name="maintenance_plan.csv",

        mime="text/csv"
    )

with open(
    "results/final_dataset.csv",
    "rb"
) as file:

    st.download_button(

        label="Download Final Dataset",

        data=file,

        file_name="final_dataset.csv",

        mime="text/csv"
    )

if (
    scenario_df is not None
    and os.path.exists(
        "results/scenario_analysis.csv"
    )
):

    with open(
        "results/scenario_analysis.csv",
        "rb"
    ) as file:

        st.download_button(

            label="Download Scenario Analysis",

            data=file,

            file_name="scenario_analysis.csv",

            mime="text/csv"
        )

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown("""
### Industrial AI Pipeline

1. Industrial sensor simulation  
2. Feature engineering  
3. Isolation Forest anomaly detection  
4. XGBoost failure prediction  
5. RUL estimation  
6. SHAP explainability  
7. MILP optimization  
8. Scenario analysis  

Built for predictive maintenance optimization.
""")