import streamlit as st
import numpy as np
import pandas as pd
import time
import requests

# Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND)
st.set_page_config(page_title="Remote Car Control", layout="wide")

# Custom CSS styling (placed after set_page_config)
st.markdown("""
<style>
:root {
    --gradient-start: #ff4444;
    --gradient-end: #4466ff;
}

.stApp {
    background: linear-gradient(to bottom, #0a0a1a, #000033);
    color: white;
    font-family: 'Arial', sans-serif;
}

h1 {
    background: linear-gradient(90deg, var(--gradient-start), var(--gradient-end));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    font-size: 2.5rem !important;
    margin: 2rem 0 !important;
    padding: 0.5rem 0 !important;
}

.stButton>button {
    background: linear-gradient(90deg, var(--gradient-start), var(--gradient-end)) !important;
    color: white !important;
    border: none !important;
    border-radius: 30px !important;
    padding: 12px 24px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    width: 100%;
    margin: 8px 0 !important;
    position: relative;
    overflow: hidden;
}

.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 15px rgba(255, 68, 68, 0.4);
}

.stImage {
    border-radius: 16px;
    border: 2px solid rgba(255,255,255,0.1);
    background: linear-gradient(145deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
    padding: 8px;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
}

[data-testid="stSidebar"] {
    background: linear-gradient(195deg, rgba(17,17,34,0.9), rgba(8,8,44,0.9)) !important;
    border-right: 1px solid rgba(255,255,255,0.1) !important;
    backdrop-filter: blur(12px) !important;
}

[data-testid="stMetric"] {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
    padding: 16px !important;
    margin: 12px 0 !important;
}

[data-testid="stLineChart"] {
    border-radius: 16px !important;
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    padding: 16px;
}

h2 {
    color: rgba(255,255,255,0.9) !important;
    border-left: 4px solid var(--gradient-start);
    padding-left: 12px !important;
    margin-top: 2rem !important;
    font-size: 1.5rem !important;
}

.st-emotion-cache-1dp5vir {
    background-image: linear-gradient(90deg, var(--gradient-start), var(--gradient-end));
}
</style>
""", unsafe_allow_html=True)

# URL of the Raspberry Pi Flask server
PI_SERVER_URL = "http://100.66.211.44:5000/command"
CAMERA_FEED_URL = "http://100.66.211.44:5000/video_feed"
CROP_DATA_URL = "http://100.66.211.44:5000/crop_data"  # Endpoint for crop data

# Simulated sensor data function
def get_sensor_data():
    return {
        "Humidity": np.random.randint(30, 80),
        "Soil Moisture": np.random.randint(200, 800),
    }

# Initialize session state for sensor data if not present
if "sensor_history" not in st.session_state:
    st.session_state.sensor_history = pd.DataFrame(columns=["Time", "Humidity", "Soil Moisture"])
if "humidity" not in st.session_state:
    st.session_state.humidity = 0
    st.session_state.soil_moisture = 0

# Initialize session state for crop data
if "crop_data" not in st.session_state:
    st.session_state.crop_data = {
        "crop": "None",
        "humidity": "0%",
        "soil_moisture": "0%"
    }

# Update sensor readings (call this once per second)
def update_sensor_data():
    new_data = get_sensor_data()
    timestamp = time.strftime("%H:%M:%S")
    # Append and keep only the latest 50 entries
    st.session_state.sensor_history = pd.concat([
        st.session_state.sensor_history,
        pd.DataFrame([[timestamp, new_data["Humidity"], new_data["Soil Moisture"]]], 
                     columns=["Time", "Humidity", "Soil Moisture"])
    ]).tail(50)
    st.session_state.humidity = new_data["Humidity"]
    st.session_state.soil_moisture = new_data["Soil Moisture"]

# Fetch crop data from Flask API
def fetch_crop_data():
    try:
        response = requests.get(CROP_DATA_URL)
        if response.status_code == 200:
            st.session_state.crop_data = response.json()
        else:
            st.error("Failed to fetch crop data from the server.")
    except requests.exceptions.RequestException:
        st.error("Could not reach the Raspberry Pi.")

# Auto-refresh sensor data and graphs every 1 second (1000 ms)
from streamlit_autorefresh import st_autorefresh
st_autorefresh(interval=1000, key="sensor_update")
update_sensor_data()
fetch_crop_data()  # Fetch crop data every second

# Sidebar - Sensor Readings
st.sidebar.header("📊 Sensor Stats (Auto-updates every 1s)")
st.sidebar.metric("Humidity (%)", st.session_state.humidity)
st.sidebar.metric("Soil Moisture", st.session_state.soil_moisture)

# Sidebar - Crop Data
st.sidebar.header("🌱 Crop Data")
st.sidebar.write(f"Detected Crop: **{st.session_state.crop_data['crop']}**")
st.sidebar.write(f"Recommended Humidity: **{st.session_state.crop_data['humidity']}**")
st.sidebar.write(f"Recommended Soil Moisture: **{st.session_state.crop_data['soil_moisture']}**")

def send_command(command):
    try:
        response = requests.post(PI_SERVER_URL, json={"command": command})
        if response.status_code == 200:
            pass
        else:
            st.error("Failed to send command.")
    except requests.exceptions.RequestException:
        st.error("Could not reach the Raspberry Pi.")

# Title
st.title("Remote Car Control Dashboard")

# Main Layout
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("📹 Live Camera Feed")
    st.markdown(f'<img src="{CAMERA_FEED_URL}" width="100%" style="border-radius:16px;">', 
                unsafe_allow_html=True)

with col2:
    st.subheader("🎮 Controls")
    if st.button("⬆️ Forward", key="forward"):
        send_command("Moving Forward")
    col_left, col_center, col_right = st.columns(3)
    with col_left:
        if st.button("⬅️ Left", key="left"):
            send_command("Turning Left")
    with col_center:
        if st.button("🌱 Soil", key="stop"):
            send_command("Lowering Sensor")
    with col_right:
        if st.button("➡️ Right", key="right"):
            send_command("Turning Right")
    if st.button("⬇️ Backward", key="backward"):
        send_command("Moving Backward")

# Live Updating Sensor Charts
st.subheader("📈 Sensor Data Over Time")
col3, col4 = st.columns(2)
with col3:
    st.line_chart(st.session_state.sensor_history.set_index("Time")["Humidity"])
with col4:
    st.line_chart(st.session_state.sensor_history.set_index("Time")["Soil Moisture"])