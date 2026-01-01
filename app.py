import streamlit as st
import joblib
import pandas as pd
import numpy as np
import zipfile
import os

# --------------------------------------------------
# Load Model (auto-extract if zipped)
# --------------------------------------------------
MODEL_ZIP = "random_forest_model_compressed.zip"
MODEL_PATH = "random_forest_model.joblib"
SCALER_PATH = "scaler.joblib"

if not os.path.exists(MODEL_PATH):
    with zipfile.ZipFile(MODEL_ZIP, "r") as zip_ref:
        zip_ref.extractall(".")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
st.set_page_config(page_title="Bike Rental Prediction", layout="centered")

st.title("🚴 Bike Rental Count Prediction")
st.write("Predict bike rental demand using Machine Learning.")

# --------------------------------------------------
# Feature Mappings
# --------------------------------------------------
season_map = {"Fall": 0, "Spring": 1, "Summer": 2, "Winter": 3}
holiday_map = {"No": 0, "Yes": 1}
workingday_map = {"No": 0, "Yes": 1}
weather_map = {"Clear": 0, "Heavy Rain": 1, "Light Snow": 2, "Mist": 3}

# --------------------------------------------------
# Input Section
# --------------------------------------------------
with st.sidebar:
    st.header("Input Parameters")

    season = st.selectbox("Season", list(season_map.keys()))
    year = st.selectbox("Year", [2011, 2012])
    month = st.slider("Month", 1, 12, 6)
    hour = st.slider("Hour", 0, 23, 12)
    holiday = st.selectbox("Holiday", list(holiday_map.keys()))
    weekday = st.slider("Weekday (0=Sun)", 0, 6, 3)
    workingday = st.selectbox("Working Day", list(workingday_map.keys()))
    weather = st.selectbox("Weather Condition", list(weather_map.keys()))
    temp = st.slider("Temperature (0–1)", 0.0, 1.0, 0.5)
    hum = st.slider("Humidity (0–1)", 0.0, 1.0, 0.6)
    wind = st.slider("Wind Speed (0–1)", 0.0, 1.0, 0.3)

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("Predict Bike Rentals"):
    input_df = pd.DataFrame([[
        season_map[season],
        year,
        month,
        hour,
        holiday_map[holiday],
        weekday,
        workingday_map[workingday],
        weather_map[weather],
        temp,
        hum,
        wind
    ]], columns=[
        "season", "yr", "mnth", "hr", "holiday",
        "weekday", "workingday", "weathersit",
        "temp", "hum", "windspeed"
    ])

    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    st.success(f"🚲 Estimated Bike Rentals: **{int(prediction)}**")

st.markdown("---")
st.caption("Model: Random Forest Regressor | ML-based Demand Prediction")
