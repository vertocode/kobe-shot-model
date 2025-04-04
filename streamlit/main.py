import streamlit as st
import pandas as pd
import requests

st.title("🏀 Kobe Bryant Shot Prediction")

with st.form("prediction_form"):
    lat = st.number_input("Latitude (lat)", min_value=-90.0, max_value=90.0, value=34.0443, format="%.4f")
    lon = st.number_input("Longitude (lon)", min_value=-180.0, max_value=180.0, value=-118.4268, format="%.4f")
    minutes_remaining = st.number_input("Minutes Remaining", min_value=0, max_value=12, value=10)
    period = st.selectbox("Period", options=list(range(1, 8)))
    playoffs = st.selectbox("Playoffs", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    shot_distance = st.number_input("Shot Distance", min_value=0, max_value=100, value=15)

    submitted = st.form_submit_button("Predict")

    if submitted:
        input_data = pd.DataFrame([{
            "lat": lat,
            "lon": lon,
            "minutes_remaining": minutes_remaining,
            "period": period,
            "playoffs": playoffs,
            "shot_distance": shot_distance
        }])

        response = requests.post(
            url="http://localhost:5050/invocations",
            headers={"Content-Type": "application/json"},
            json={"dataframe_split": input_data.to_dict(orient="split")}
        )

        if response.status_code == 200:
            prediction = response.json()["predictions"][0]
            result_text = "✅ Kobe made the shot!" if prediction == 1 else "❌ Kobe missed the shot."
            st.success(result_text)
        else:
            st.error(f"Prediction failed. Status code: {response.status_code}")