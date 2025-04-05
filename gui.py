import os
import wave
import requests
import pyaudio
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ======================= CONFIG ==========================
API_KEY = os.getenv("GOOGLE-API-KEY")
AUDIO_FILENAME = "recorded_audio.wav"
GEN_MODEL = "gemini-2.0-flash"
SYMPTOM_API_URL = "http://localhost:8000/predict/symptoms"
TREATMENT_API_URL = "http://localhost:8000/predict/treatments"

# Audio recording config
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 5

# Gemini setup
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(GEN_MODEL)

# ======================= Functions ==========================
def record_audio():
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = [stream.read(CHUNK) for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS))]
    stream.stop_stream()
    stream.close()
    audio.terminate()

    with wave.open(AUDIO_FILENAME, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

def transcribe_audio():
    if not os.path.exists(AUDIO_FILENAME):
        return "Audio file missing!"

    with open(AUDIO_FILENAME, "rb") as f:
        audio_data = {
            "mime_type": "audio/wav",
            "data": f.read()
        }

    prompt = {"text": "Extract out the disease stated in the audio"}
    response = model.generate_content(contents=[prompt, audio_data])

    return response.text.strip() if response and hasattr(response, "text") else "❌ Transcription failed."

def send_symptom_query(disease_name):
    try:
        response = requests.post(SYMPTOM_API_URL, json={"disease": disease_name})
        if response.status_code == 200:
            return response.json().get("response", "No symptom response found.")
        return f"FastAPI Error {response.status_code}:\n{response.text}"
    except requests.exceptions.RequestException as e:
        return f"❌ Request failed: {e}"

def send_treatment_query(disease_name):
    try:
        response = requests.post(TREATMENT_API_URL, json={"disease": disease_name})
        if response.status_code == 200:
            return response.json().get("treatments", "No treatments found.")
        return f"FastAPI Error {response.status_code}:\n{response.text}"
    except requests.exceptions.RequestException as e:
        return f"❌ Request failed: {e}"

# ======================= Streamlit UI ==========================

st.set_page_config(page_title="🎙️ GPT-2 Medi", layout="centered")
st.title("🎙️ GPT-2 Medi Voice Diagnosis")

if st.button("Start Voice Diagnosis 🎤"):
    with st.spinner("Recording audio..."):
        record_audio()

    with st.spinner("Transcribing disease from audio..."):
        disease = transcribe_audio()

    st.subheader("📄 Transcription")
    st.text_area("Disease Mentioned", disease, height=100)

    with st.spinner("Fetching symptoms..."):
        symptoms = send_symptom_query(disease)

    st.subheader("🧬 Symptoms Analysis")
    st.text_area("Symptoms", symptoms, height=150)

    with st.spinner("Fetching treatments..."):
        treatments = send_treatment_query(disease)

    st.subheader("💊 Treatment Recommendations")
    st.text_area("Treatments", treatments, height=150)

st.markdown("---")
st.caption("Made using Streamlit and FastAPI")
st.caption("© 2025 Shivansh Tyagi")
st.caption("All rights reserved.")
