import streamlit as st
import serial
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks, savgol_filter
from collections import deque
import requests
import math
import json
import os

ESP32_URL = "http://192.168.29.197"  # Replace with your ESP32's IP
TOUCH_THRESHOLD = 100000
READ_DURATION = 30  # seconds


# === CONFIGURATION ===
COM_PORT = '/dev/cu.usbserial-0001'  # Change as needed
BAUD_RATE = 115200
READ_DURATION = 30  # seconds
TOUCH_THRESHOLD = 100000  # IR threshold to detect finger touch
SAVE_IR_PATH = 'ir_cleaned_data.csv'
SAVE_RED_PATH = 'red_cleaned_data.csv'

def send_to_thingspeak(spo2, rr, hr, name, age, gender, api_key):
    url = "https://api.thingspeak.com/update"
    payload = {
        "api_key": api_key,
        "field1": rr,
        "field2": spo2,
        "field3": hr,
        "field4": age,
        "field5": gender,
        "field6": name,
    }

    try:
        response = requests.get(url, params=payload)
        if response.status_code == 200:
            st.success("📡 Data pushed to ThingSpeak!")
        else:
            st.error(f"❌ ThingSpeak push failed. Status: {response.status_code}")
    except Exception as e:
        st.error(f"❌ Error sending to ThingSpeak: {e}")

# === FILTERING ===
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return filtfilt(b, a, data)

def preprocess_signal(signal, fs):
    bandpassed = bandpass_filter(signal, 0.1, 0.5, fs)
    smoothed = savgol_filter(bandpassed, window_length=51, polyorder=3)
    smoothed += np.mean(signal)  # Shift back to positive range
    return smoothed

def detect_breath_peaks(signal, fs):
    min_interval = int(1.5 * fs)
    peaks, _ = find_peaks(signal, distance=min_interval, prominence=0.05)
    return peaks

# === SpO2 Calculation ===
def calculate_spo2(ir_raw, red_raw, fs):
    # Use bandpass filter from 0.5 to 3 Hz for SpO2 pulse detection
    ir_filtered = bandpass_filter(ir_raw, 0.5, 3.0, fs)
    red_filtered = bandpass_filter(red_raw, 0.5, 3.0, fs)

    peaks, _ = find_peaks(ir_filtered, distance=int(0.6 * fs), prominence=0.02 * max(ir_filtered))
    R_values = []

    for i in range(len(peaks) - 1):
        start = peaks[i]
        end = peaks[i + 1]

        if end - start < 3:
            continue

        ir_seg = ir_raw[start:end]
        red_seg = red_raw[start:end]
        ir_filt_seg = ir_filtered[start:end]
        red_filt_seg = red_filtered[start:end]

        AC_ir = np.max(ir_filt_seg) - np.min(ir_filt_seg)
        DC_ir = np.mean(ir_seg)
        AC_red = np.max(red_filt_seg) - np.min(red_filt_seg)
        DC_red = np.mean(red_seg)

        if DC_ir == 0 or DC_red == 0:
            continue

        R = (AC_red / DC_red) / (AC_ir / DC_ir)
        R_values.append(R)

    if not R_values:
        return 0.0, []

    R_avg = np.mean(R_values)
    spo2 = 104 - 17 * R_avg
    spo2 = max(0, min(100, spo2))

    return spo2, R_values

# === Heart Rate ===
def calculate_heart_rate(ir_values, fs):
    ir_filtered = bandpass_filter(ir_values, 0.8, 2.5, fs)  # Heart rate: ~48–150 BPM
    peaks, _ = find_peaks(ir_filtered, distance=int(0.5 * fs), prominence=0.4 * np.std(ir_filtered))

    duration_sec = len(ir_filtered) / fs
    heart_rate = (len(peaks) / duration_sec) * 60  # Convert to BPM
    heart_rate = round(heart_rate)

    return heart_rate, ir_filtered, peaks

# === IR AND RED DATA COLLECTION ===


def read_ir_data():
    st.write(f"Connecting to ESP32 at {ESP32_URL}...")

    ir_values = []
    red_values = []
    touched = False
    start_time = None

    timer_placeholder = st.empty()
    ydata_ir = deque([0] * 200, maxlen=200)
    ydata_red = deque([0] * 200, maxlen=200)

    st.write("Waiting for finger....")

    try:
        while True:
            try:
                response = requests.get(f"{ESP32_URL}/getdata", timeout=1)
                line_serial = response.text.strip()
            except Exception as e:
                st.error(f"Connection error: {e}")
                continue

            if "IR[" in line_serial and "RED[" in line_serial:
                try:
                    ir = int(line_serial.split("IR[")[1].split("]")[0])
                    red = int(line_serial.split("RED[")[1].split("]")[0])
                except:
                    continue

                if not touched and ir > TOUCH_THRESHOLD:
                    st.write("Touch detected. Starting data collection...")
                    touched = True
                    start_time = time.time()

                if touched:
                    elapsed = time.time() - start_time
                    time_left = max(0, int(READ_DURATION - elapsed))
                    timer_placeholder.markdown(f"⏳ Time left: **{time_left} seconds**")

                    ir_values.append(ir)
                    red_values.append(red)
                    ydata_ir.append(ir)
                    ydata_red.append(red)

                    if elapsed > READ_DURATION:
                        st.success("✅ Data collection complete.")
                        break
    finally:
        st.write("Stopped data fetching.")

    timer_placeholder.empty()
    return ir_values, red_values


def read_ir_data_serial():
    st.write(f"Connecting to {COM_PORT}...")
    ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)  # wait for serial connection to settle
    ir_values = []
    red_values = []
    touched = False
    start_time = None

    timer_placeholder = st.empty()  # placeholder for the timer display

    # Deques retained for potential offline plotting
    ydata_ir = deque([0] * 200, maxlen=200)
    ydata_red = deque([0] * 200, maxlen=200)

    # Plot areas disabled
    # plot_area_ir = st.empty()
    # plot_area_red = st.empty()

    st.write("Waiting for finger....")

    try:
        while True:
            line_serial = ser.readline().decode(errors='ignore').strip()

            if "IR[" in line_serial and "RED[" in line_serial:
                try:
                    ir = int(line_serial.split("IR[")[1].split("]")[0])
                    red = int(line_serial.split("RED[")[1].split("]")[0])
                except:
                    continue

                if not touched and ir > TOUCH_THRESHOLD:
                    st.write("Touch detected. Starting data collection...")
                    touched = True
                    start_time = time.time()

                if touched:
                    elapsed = time.time() - start_time
                    time_left = max(0, int(READ_DURATION - elapsed))
                    timer_placeholder.markdown(f"⏳ Time left: **{time_left} seconds**")

                    ir_values.append(ir)
                    red_values.append(red)
                    ydata_ir.append(ir)
                    ydata_red.append(red)

                    if elapsed > READ_DURATION:
                        st.success("✅ Data collection complete.")
                        break
    finally:
        ser.close()
        st.write("Serial port closed.")

    timer_placeholder.empty()  # clear timer after done
    return ir_values, red_values

# === SIGNAL ANALYSIS ===
def analyze_signal(ir_values, red_values, fs, name, age, gender):
    st.subheader("Respiratory Signal Analysis")
    filtered_ir = preprocess_signal(ir_values, fs)
    peaks = detect_breath_peaks(filtered_ir, fs)
    rr = math.ceil(len(peaks) * (60 / READ_DURATION))

    st.success(f"✅ Detected {len(peaks)} breaths in {READ_DURATION} sec → Estimated RR: {rr:.2f}")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(filtered_ir, label='Filtered IR Signal')
    ax.plot(peaks, np.array(filtered_ir)[peaks], 'ro', label='Detected Breaths')
    ax.axhline(np.mean(filtered_ir), color='gray', linestyle='--', label='Mean Line')
    ax.set_title(f'Respiratory Signal with Peaks — Rate: {rr:.2f} RR')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Filtered IR Value')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    st.subheader("Heart Rate Estimation")
    hr, hr_filtered, hr_peaks = calculate_heart_rate(ir_values, fs)
    st.success(f"❤️ Estimated Heart Rate: {hr} BPM")

    fig_hr, ax_hr = plt.subplots(figsize=(12, 5))
    ax_hr.plot(hr_filtered, label='Filtered IR (Heart)', alpha=0.8)
    ax_hr.plot(hr_peaks, hr_filtered[hr_peaks], 'rx', label='Detected Beats')
    ax_hr.set_title(f"Heart Rate Signal — HR: {hr} BPM")
    ax_hr.set_xlabel("Sample Index")
    ax_hr.set_ylabel("Filtered IR (Pulse)")
    ax_hr.legend()
    ax_hr.grid(True)
    st.pyplot(fig_hr)

    # SpO2 Calculation & Display
    st.subheader("SpO₂ Estimation")
    spo2, R_list = calculate_spo2(np.array(ir_values), np.array(red_values), fs)
    if spo2 > 0:
        st.success(f"✅ Estimated SpO₂: {spo2:.2f}% using {len(R_list)} pulse cycles.")
    else:
        st.warning("⚠ Unable to estimate SpO₂ reliably.")

    # Optional plot of raw signals with SpO2
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.plot(ir_values, label='IR Raw', alpha=0.6)
    ax2.plot(red_values, label='RED Raw', alpha=0.6)
    ax2.set_title(f"Raw Signals — Estimated SpO₂ = {spo2:.2f}%")
    ax2.set_xlabel("Sample Index")
    ax2.set_ylabel("ADC Value")
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)
    
    THINGSPEAK_API_KEY = "VX26ZPD2D2YK5JRJ"  # Replace with your ThingSpeak write API key
    send_to_thingspeak(spo2, rr, hr, name, age, gender, THINGSPEAK_API_KEY)

# === MAIN APP ===
def main():
    st.title("Respiratory, Heart Rate & SpO₂ Estimator")

    st.sidebar.header("👤 Patient Info")
    name = st.sidebar.text_input("Name")
    age = st.sidebar.text_input("Age")
    gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
    mode = st.sidebar.selectbox("Data Mode", ["Wi-Fi (ESP32)", "Serial (USB)"])

    # if mode == "Wi-Fi (ESP32)":
    #     ir_values, red_values = read_ir_data()  # new Wi-Fi version
    # else:
    #     r_values, red_values = read_ir_data_serial()  # keep your old serial logic under this name

    if st.button("Start Data Collection"):
        ir_values, red_values = read_ir_data()
        if not ir_values or not red_values:
            st.warning("⚠ No IR or RED data collected.")
            return

        fs = len(ir_values) / READ_DURATION
        # fs=20

        # Save CSVs
        pd.DataFrame(ir_values, columns=["IR"]).to_csv(SAVE_IR_PATH, index=False)
        pd.DataFrame(red_values, columns=["RED"]).to_csv(SAVE_RED_PATH, index=False)
        st.write(f"📁 Saved {len(ir_values)} IR samples to {SAVE_IR_PATH}")
        st.write(f"📁 Saved {len(red_values)} RED samples to {SAVE_RED_PATH}")

        analyze_signal(ir_values, red_values, fs, name, age, gender)

if __name__ == "__main__":
    main()