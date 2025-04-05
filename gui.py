import os
import wave
import threading
import pyaudio
import requests
import google.generativeai as genai
import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
from dotenv import load_dotenv
load_dotenv()


# ======================= CONFIG ==========================
API_KEY = os.getenv("GOOGLE-API-KEY")
AUDIO_FILENAME = "recorded_audio.wav"
GEN_MODEL = "gemini-2.0-flash"
SYMPTOM_API_URL = "http://localhost:8000/predict/symptoms"
TREATMENT_API_URL = "http://localhost:8000/predict/treatments"

# Audio recording
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 5

# Gemini Config
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(GEN_MODEL)


def record_audio(status_label):
    try:
        status_label.config(text="🎙️ Recording...")
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

        status_label.config(text="✅ Recording complete.")
    except Exception as e:
        status_label.config(text=f"❌ Recording error: {e}")

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



def run_pipeline(status_label, transcript_box, response_box, treatment_box):
    record_audio(status_label)
    status_label.config(text="🤖 Transcribing...")
    disease = transcribe_audio()

    transcript_box.delete("1.0", tk.END)
    transcript_box.insert(tk.END, disease)

    status_label.config(text="📡 Fetching symptoms...")
    symptoms = send_symptom_query(disease)

    status_label.config(text="📡 Fetching treatments...")
    treatments = send_treatment_query(disease)

    response_box.delete("1.0", tk.END)
    response_box.insert(tk.END, symptoms)

    treatment_box.delete("1.0", tk.END)
    treatment_box.insert(tk.END, treatments)

    status_label.config(text="✅ Done.")


def start_process(status_label, transcript_box, response_box, treatment_box):
    threading.Thread(target=run_pipeline, args=(status_label, transcript_box, response_box, treatment_box)).start()

# ======================= GUI SETUP ==========================
root = tk.Tk()
root.title("Voice Medical Assistant")
root.geometry("720x700")
root.configure(bg="#eaf0f6")

style = ttk.Style()
style.theme_use("clam")
style.configure("TButton",
                font=("Segoe UI", 12, "bold"),
                padding=10,
                relief="flat",
                background="#5c4dff",
                foreground="white")
style.map("TButton",
          background=[('active', '#4b39db')],
          relief=[('pressed', 'sunken')])

# Fonts
TITLE_FONT = ("Segoe UI", 20, "bold")
LABEL_FONT = ("Segoe UI", 12, "bold")
BOX_FONT = ("Segoe UI", 11)

# Title
title_label = ttk.Label(root, text="🎙️ Voice Medical Assistant", font=TITLE_FONT, background="#eaf0f6", foreground="#111827")
title_label.pack(pady=(30, 10))

# Status label
status_label = ttk.Label(root, text="Ready", font=LABEL_FONT, background="#eaf0f6", foreground="#374151")
status_label.pack(pady=(0, 20))

# Main container frame
container = tk.Frame(root, bg="white", bd=0, relief="flat")
container.pack(padx=40, pady=10, fill="both", expand=True)

# Start Diagnosis Button
start_btn = ttk.Button(container, text="🎙️ Start Voice Diagnosis",
                       command=lambda: start_process(status_label, transcript_box, response_box, treatment_box))
start_btn.pack(fill="x", pady=(10, 20), padx=20)

# Box Styles
box_config = {
    "wrap": tk.WORD,
    "height": 5,
    "font": BOX_FONT,
    "bg": "#f9fafb",
    "bd": 1,
    "relief": "solid"
}

# Transcription Box
tk.Label(container, text="📄 Transcription", bg="white", font=LABEL_FONT, anchor="w").pack(anchor="w", padx=20, pady=(0, 5))
transcript_box = scrolledtext.ScrolledText(container, **box_config)
transcript_box.pack(padx=20, pady=(0, 20), fill="x")

# Symptoms Analysis Box
tk.Label(container, text="🧬 Symptoms Analysis", bg="white", font=LABEL_FONT, anchor="w").pack(anchor="w", padx=20, pady=(0, 5))
box_config["height"] = 6
response_box = scrolledtext.ScrolledText(container, **box_config)
response_box.pack(padx=20, pady=(0, 20), fill="x")

# Treatment Recommendations Box
tk.Label(container, text="💊 Treatment Recommendations", bg="white", font=LABEL_FONT, anchor="w").pack(anchor="w", padx=20, pady=(0, 5))
box_config["height"] = 6
treatment_box = scrolledtext.ScrolledText(container, **box_config)
treatment_box.pack(padx=20, pady=(0, 20), fill="x")

# Exit Button
exit_btn = ttk.Button(root, text="❌ Close Application", command=root.quit)
exit_btn.pack(pady=(10, 20))

root.mainloop()
