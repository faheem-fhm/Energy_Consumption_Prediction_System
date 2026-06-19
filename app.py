import streamlit as st
import pandas as pd
import pickle

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="Energy Consumption Prediction",
    page_icon="",
    layout="wide"
)

# -------------------------
# Custom CSS
# -------------------------
st.markdown("""
<style>
.main {
    padding-top: 1rem;
}
.stMetric {
    background-color: #f8f9fa;
    padding: 10px;
    border-radius: 10px;
}
h1, h2, h3 {
    color: #1f77b4;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Load Model
# -------------------------
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# -------------------------
# Header
# -------------------------
st.title("Energy Consumption Prediction System")
st.markdown("### Predict Household Power Consumption Using Machine Learning")

# -------------------------
# Input Section
# -------------------------
col1, col2 = st.columns(2)

with col1:
    global_reactive_power = st.number_input(
        "Global Reactive Power",
        min_value=0.0,
        value=0.10
    )

    voltage = st.number_input(
        "Voltage",
        min_value=0.0,
        value=240.0
    )

    global_intensity = st.number_input(
        "Global Intensity",
        min_value=0.0,
        value=10.0
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

# -------------------------
# Derived Features
# -------------------------
total_sub_metering = (
    sub_metering_1 +
    sub_metering_2 +
    sub_metering_3
)

apparent_power = voltage * global_intensity
st.subheader("Energy Metrics")

m1, m2 = st.columns(2)

m1.metric(
    "Total Sub Metering",
    total_sub_metering
)

m2.metric(
    "Estimated Apparent Power",
    f"{apparent_power:.2f}"
)

# -------------------------
# Prediction
# -------------------------
if st.button(" Predict Energy Consumption",
             use_container_width=True):

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
        f" Predicted Energy Consumption: {prediction:.3f} kW"
    )

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.markdown(
    "Developed by **Mohamed Faheem** | Energy Analytics Dashboard"
)