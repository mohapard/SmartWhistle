import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import numpy as np
import openai
import librosa, pickle, os, shutil, wave
from sklearn.linear_model import LogisticRegression
from datetime import datetime
import time
import matplotlib.pyplot as plt
import librosa.display

# --- CONFIG ---
openai.api_key = st.secrets["openai_key"]  # Use Streamlit Secrets for your API key
MODEL_FILE = "whistle_mel_model.pkl"
DATASET_DIR = "whistle_dataset"
WHISTLE_DURATION = 3  # seconds
os.makedirs(DATASET_DIR, exist_ok=True)

# --- SESSION STATE ---
if "events" not in st.session_state: st.session_state.events = []
if "last_transcription" not in st.session_state: st.session_state.last_transcription = ""
if "labels" not in st.session_state: st.session_state.labels = {"single": [], "double": []}

# --- Helper: Save audio frames to WAV ---
def save_audio(frames, filename="sample.wav", rate=44100):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))

# --- Feature Extraction ---
def extract_mel(filename):
    y, sr = librosa.load(filename, sr=None)
    y = librosa.util.fix_length(y, size=sr*WHISTLE_DURATION)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.flatten()

# --- Train Classifier ---
def train_classifier(single_whistles, double_whistles):
    X = []
    y = []
    for f in single_whistles:
        X.append(extract_mel(f)); y.append(0)
    for f in double_whistles:
        X.append(extract_mel(f)); y.append(1)
    model = LogisticRegression(max_iter=500)
    model.fit(X, y)
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    return model

# --- Load model if exists ---
def load_model():
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "rb") as f:
            return pickle.load(f)
    return None

# --- Predict whistle type ---
def predict_whistle(filename):
    model = load_model()
    if model is None:
        return "Model not trained"
    features = extract_mel(filename).reshape(1, -1)
    prediction = model.predict(features)[0]
    return "Single" if prediction == 0 else "Double"

# --- WebRTC audio processor ---
class AudioProcessor:
    def __init__(self):
        self.frames = []
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        pcm = frame.to_ndarray().tobytes()
        self.frames.append(pcm)
        return frame

# --- UI ---
st.title("Smart Whistle Recorder & Analyzer")

st.write("### Record whistle via browser:")
webrtc_ctx = webrtc_streamer(
    key="audio_recorder",
    mode=WebRtcMode.SENDRECV,
    audio_receiver_size=256,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True,
    audio_processor_factory=AudioProcessor,
)

if webrtc_ctx.audio_receiver:
    if st.button("Save & Analyze Whistle"):
        processor = webrtc_ctx.audio_processor
        if processor and processor.frames:
            filename = "sample.wav"
            save_audio(processor.frames, filename)
            st.success(f"Audio saved as {filename} ({len(processor.frames)} frames)")

            # Predict whistle type
            result = predict_whistle(filename)
            st.write(f"### Predicted whistle type: **{result}**")

            # Show spectrogram
            mel = extract_mel(filename).reshape(-1, 128).T
            fig, ax = plt.subplots()
            librosa.display.specshow(mel, x_axis='time', y_axis='mel', sr=44100, ax=ax)
            ax.set_title("Mel Spectrogram")
            st.pyplot(fig)
        else:
            st.warning("No audio captured yet.")

# --- Training section ---
st.write("### Train model with new data")
label = st.selectbox("Label this whistle", ["single", "double"])
if st.button("Save sample to dataset"):
    if os.path.exists("sample.wav"):
        dest = os.path.join(DATASET_DIR, f"{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav")
        shutil.copy("sample.wav", dest)
        st.session_state.labels[label].append(dest)
        st.success(f"Saved sample as {dest}")
    else:
        st.warning("No sample to save.")

if st.button("Train model"):
    singles = st.session_state.labels["single"]
    doubles = st.session_state.labels["double"]
    if singles and doubles:
        train_classifier(singles, doubles)
        st.success("Model trained and saved.")
    else:
        st.warning("Need samples for both single and double whistles.")
