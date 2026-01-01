Python 3.14.2 (v3.14.2:df793163d58, Dec  5 2025, 12:18:06) [Clang 16.0.0 (clang-1600.0.26.6)] on darwin
Enter "help" below or click "Help" above for more information.
>>> import streamlit as st
... import joblib
... import pandas as pd
... import numpy as np
... 
... # --------------------------------------------------
... # Load Model and Scaler
... # --------------------------------------------------
... model = joblib.load("random_forest_model.joblib")
... scaler = joblib.load("scaler.joblib")
... 
... # --------------------------------------------------
... # App UI
... # --------------------------------------------------
... st.set_page_config(page_title="Bike Rental Prediction", layout="centered")
... 
... st.title("🚲 Bike Rental Count Prediction")
... st.write("Enter the details below to predict the number of bike rentals.")
... 
... # --------------------------------------------------
... # Mappings (Must match training phase)
... # --------------------------------------------------
... season_mapping = {
...     "Fall": 0,
...     "Spring": 1,
...     "Summer": 2,
...     "Winter": 3
... }
... 
... holiday_mapping = {
...     "No": 0,
...     "Yes": 1
... }
... 
... workingday_mapping = {
...     "No": 0,
...     "Yes": 1
}

weathersit_mapping = {
    "Clear": 0,
    "Heavy Rain": 1,
    "Light Snow": 2,
    "Mist": 3
}

# --------------------------------------------------
# Sidebar Inputs
# --------------------------------------------------
with st.sidebar:
    st.header("Input Features")

    season = st.selectbox("Season", list(season_mapping.keys()))
    yr = st.selectbox("Year", [2011, 2012])
    mnth = st.slider("Month", 1, 12, 7)
    hr = st.slider("Hour", 0, 23, 12)
    holiday = st.selectbox("Holiday", list(holiday_mapping.keys()))
    weekday = st.slider("Weekday (0=Sun, 6=Sat)", 0, 6, 3)
    workingday = st.selectbox("Working Day", list(workingday_mapping.keys()))
    weathersit = st.selectbox("Weather Situation", list(weathersit_mapping.keys()))
    temp = st.slider("Temperature (0–1)", 0.0, 1.0, 0.5)
    hum = st.slider("Humidity (0–1)", 0.0, 1.0, 0.6)
    windspeed = st.slider("Windspeed (0–1)", 0.0, 1.0, 0.3)

# --------------------------------------------------
# Prepare Input
# --------------------------------------------------
input_df = pd.DataFrame([[
    season_mapping[season],
    float(yr),
    float(mnth),
    hr,
    holiday_mapping[holiday],
    weekday,
    workingday_mapping[workingday],
    weathersit_mapping[weathersit],
    temp,
    hum,
    windspeed
]], columns=[
    "season", "yr", "mnth", "hr", "holiday",
    "weekday", "workingday", "weathersit",
    "temp", "hum", "windspeed"
])

# Scale input
scaled_input = scaler.transform(input_df)

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("Predict Bike Rental Count"):
    prediction = model.predict(scaled_input)
    prediction = max(0, int(np.round(prediction[0])))

    st.success(f"🚴 Predicted Bike Rental Count: **{prediction}**")

st.markdown("---")
st.caption("Model: Random Forest Regressor | Features normalized as per training data")
