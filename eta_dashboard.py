# your streamlit app code (paste eta_dashboard.py here)

import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import time
from datetime import datetime, timedelta
from google.transit import gtfs_realtime_pb2
from tensorflow.keras.models import load_model
import folium
from streamlit_folium import st_folium
import gtfs_realtime_pb2



# === Load Model and Encoders ===
model = load_model("lstm_eta_model.h5")
scaler = joblib.load("feature_scaler.pkl")
weather_encoder = joblib.load("weather_encoder.pkl")

# === API KEYS ===
MTA_API_KEY = "bab3392b-58f0-42c2-8b61-421d6a03e72e"
TOMTOM_API_KEY = "gmKSHRhMEQ1oXOnhV5wKL2B3WE45SZL9"
OPENWEATHER_API_KEY = "d7836e8948f06edd3c191fa978ff266f"

# === API URLs ===
MTA_API_URL = "https://gtfsrt.prod.obanyc.com/vehiclePositions"
TOMTOM_URL = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
OPENWEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"

# === Helper Functions ===
def fetch_mta_data():
    headers = {"x-api-key": MTA_API_KEY}
    response = requests.get(MTA_API_URL, headers=headers)
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(response.content)
    buses = []
    for entity in feed.entity:
        if entity.HasField("vehicle"):
            v = entity.vehicle
            buses.append({
                "vehicle_id": v.vehicle.id,
                "route_id": v.trip.route_id,
                "trip_id": v.trip.trip_id,
                "latitude": v.position.latitude,
                "longitude": v.position.longitude,
                "timestamp": v.timestamp
            })
    return buses[:10]  # Limit to 10 buses for demo

def fetch_traffic(lat, lon):
    params = {"point": f"{lat},{lon}", "unit": "KMPH", "key": TOMTOM_API_KEY}
    r = requests.get(TOMTOM_URL, params=params)
    if r.status_code == 200:
        d = r.json()
        return round(d["flowSegmentData"]["currentTravelTime"] / d["flowSegmentData"]["freeFlowTravelTime"], 2)
    return 1.0

def fetch_weather(lat, lon):
    params = {"lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY, "units": "metric"}
    r = requests.get(OPENWEATHER_URL, params=params)
    if r.status_code == 200:
        d = r.json()
        return d["main"]["temp"], d["weather"][0]["main"]
    return 25.0, "Clear"

def convert_to_ist(utc_timestamp):
    utc_time = datetime.utcfromtimestamp(utc_timestamp)
    ist_time = utc_time + timedelta(hours=5, minutes=30)
    return ist_time.strftime("%H:%M:%S")

# === Streamlit UI ===
st.set_page_config(page_title="Bus ETA Live Tracker", layout="wide")
st.title("🚌 Real-Time Bus ETA Prediction (LSTM Model)")

bus_data = fetch_mta_data()
history = []

# === Map setup ===
if bus_data:
    first = bus_data[0]
    m = folium.Map(location=[first["latitude"], first["longitude"]], zoom_start=11)

    table_data = []

    for bus in bus_data:
        lat, lon = bus["latitude"], bus["longitude"]
        traffic_ratio = fetch_traffic(lat, lon)
        temp, weather = fetch_weather(lat, lon)
        ts = bus["timestamp"]
        ist_time = convert_to_ist(ts)

        if weather in weather_encoder.classes_:
            weather_code = weather_encoder.transform([weather])[0]
        else:
            weather_code = 0

        # Sequence history per bus (5 points — simulate with last value repeated)
        point = [traffic_ratio, temp, weather_code]
        history = [point] * 5
        X = np.array(history)
        X_scaled = scaler.transform(X).reshape(1, 5, 3)

        eta = float(model.predict(X_scaled)[0][0])
        eta = max(0, round(eta))

        # Add marker to map
        popup = f"Bus ID: {bus['vehicle_id']}<br>Delay: {eta} sec<br>Weather: {weather}<br>Traffic: {traffic_ratio}"
        folium.Marker([lat, lon], tooltip=f"{bus['vehicle_id']}", popup=popup,
                      icon=folium.Icon(color="blue", icon="bus", prefix="fa")).add_to(m)

        # Add to table
        table_data.append({
            "Bus ID": bus["vehicle_id"],
            "Route": bus["route_id"],
            "Time": ist_time,
            "ETA Delay (sec)": eta,
            "Traffic Ratio": traffic_ratio,
            "Temperature (°C)": temp,
            "Weather": weather
        })

    # Render map
    st_folium(m, width=700, height=500)

    # Show table
    st.subheader("📊 Live ETA Predictions")
    st.dataframe(pd.DataFrame(table_data))
else:
    st.warning("No bus data available.")
