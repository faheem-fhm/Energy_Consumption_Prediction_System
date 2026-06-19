import streamlit as st
import pandas as pd
import pickle

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Energy Consumption Prediction",
    page_icon="",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

[data-testid="stMetric"] {
    background-color: rgba(28,131,225,0.1);
    border: 1px solid rgba(28,131,225,0.3);
    padding: 15px;
    border-radius: 12px;
    text-align: center;
}

.stButton > button {
    width: 100%;
    height: 50px;
    font-size: 18px;
    font-weight: bold;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as file:
        return pickle.load(file)

model = load_model()

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.title("Energy Consumption Prediction System")
st.markdown(
    "Predict household electricity consumption using Machine Learning."
)

st.markdown("---")

# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    global_reactive_power = st.number_input(
        "Global Reactive Power",
        min_value=0.0,
        value=0.10,
        step=0.01
    )

    voltage = st.number_input(
        "Voltage",
        min_value=0.0,
        value=240.0,
        step=0.1
    )

    global_intensity = st.number_input(
        "Global Intensity",
        min_value=0.0,
        value=10.0,
        step=0.1
    )

with col2:
    sub_metering_1 = st.number_input(
        "Sub Metering 1",
        min_value=0,
        value=0
    )

    sub_metering_2 = st.number_input(
        "Sub Metering 2",
        min_value=0,
        value=1
    )

    sub_metering_3 = st.number_input(
        "Sub Metering 3",
        min_value=0,
        value=17
    )

# --------------------------------------------------
# DERIVED FEATURES
# --------------------------------------------------
total_sub_metering = (
    sub_metering_1 +
    sub_metering_2 +
    sub_metering_3
)

apparent_power = voltage * global_intensity

# --------------------------------------------------
# KPI SECTION
# --------------------------------------------------
st.subheader("Energy Metrics")

m1, m2 = st.columns(2)

with m1:
    st.metric(
        "🔌 Total Sub Metering",
        total_sub_metering
    )

with m2:
    st.metric(
        "Apparent Power",
        f"{apparent_power:.2f}"
    )

st.markdown("---")

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
if st.button("Predict Energy Consumption"):

    input_data = pd.DataFrame([{
        "Global_reactive_power": global_reactive_power,
        "Voltage": voltage,
        "Global_intensity": global_intensity,
        "Sub_metering_1": sub_metering_1,
        "Sub_metering_2": sub_metering_2,
        "Sub_metering_3": sub_metering_3,
        "Total_Sub_Metering": total_sub_metering
    }])

    prediction = model.predict(input_data)[0]

    st.success(
        f"Predicted Energy Consumption: {prediction:.3f} kW"
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.caption(
    "Developed by Mohamed Faheem | Energy Consumption Prediction Dashboard"
)
