import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av, wave, os, pickle, time
import numpy as np
import openai, librosa
from sklearn.linear_model import LogisticRegression
from datetime import datetime

# --- CONFIG ---
openai.api_key = st.secrets["openai_key"]
MODEL_FILE = "whistle_mel_model.pkl"
WHISTLE_DURATION = 3
NOTE_DURATION = 6
RATE = 44100
os.makedirs("data", exist_ok=True)

RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {
            "urls": ["turn:openrelay.metered.ca:80", "turn:openrelay.metered.ca:443"],
            "username": "openrelayproject",
            "credential": "openrelayproject"
        }
    ]
})

# --- AUDIO HELPERS ---
def save_audio(frames, filename, rate=RATE):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))
    print(f"[DEBUG] Saved {filename} ({os.path.getsize(filename)} bytes)")

def extract_mel(filename):
    y, sr = librosa.load(filename, sr=22050)
    y = librosa.util.fix_length(y, size=sr * WHISTLE_DURATION)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
    return librosa.power_to_db(mel, ref=np.max).flatten()

# --- MODEL FUNCTIONS ---
def load_model():
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "rb") as f: 
            model = pickle.load(f)
            print("[DEBUG] Model loaded")
            return model
    print("[DEBUG] No model found")
    return None

def predict_whistle(filename):
    model = load_model()
    if model is None:
        return "Unknown", 0.0
    features = extract_mel(filename).reshape(1, -1)
    proba = model.predict_proba(features)[0]
    pred_class = np.argmax(proba)
    label = "Single" if pred_class == 0 else "Double"
    confidence = proba[pred_class]
    print(f"[DEBUG] Predicted {label} with {confidence:.2f}")
    return label, confidence

def retrain_model(single_files, double_files):
    X, y = [], []
    for f in single_files:
        X.append(extract_mel(f)); y.append(0)
    for f in double_files:
        X.append(extract_mel(f)); y.append(1)
    if not X:
        return None
    model = LogisticRegression(max_iter=500)
    model.fit(X, y)
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    print("[DEBUG] Model retrained and saved")
    return model

# --- TRANSCRIPTION ---
def transcribe_audio(filename):
    print(f"[DEBUG] Transcribing {filename} using Whisper-large...")
    with open(filename, "rb") as f:
        transcript = openai.audio.transcriptions.create(
            model="whisper-1", 
            file=f
        )
    return transcript.text

# --- AUDIO PROCESSOR ---
class AudioProcessor:
    def __init__(self):
        self.frames = []
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        self.frames.append(frame.to_ndarray().astype(np.int16).tobytes())
        return frame

# --- UI ---
st.title("Smart Whistle Logger (Retrainable)")
status_box = st.empty()
progress = st.progress(0)

webrtc_ctx = webrtc_streamer(
    key="recorder",
    mode=WebRtcMode.SENDONLY,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True,
    audio_processor_factory=AudioProcessor,
    sendback_audio=False
)

if "events" not in st.session_state: 
    st.session_state.events = []

# --- MAIN FLOW ---
if webrtc_ctx.audio_processor and st.button("Log Event"):
    processor = webrtc_ctx.audio_processor
    processor.frames.clear()
    status_box.info("Blow your whistle now!")
    frames = []
    start = time.time()
    # --- Record Whistle ---
    while time.time() - start < WHISTLE_DURATION:
        if processor.frames:
            frames.extend(processor.frames)
            processor.frames.clear()
        progress.progress(int(((time.time() - start) / WHISTLE_DURATION) * 100))
        time.sleep(0.05)
    save_audio(frames, "whistle.wav")
    whistle_type, whistle_conf = predict_whistle("whistle.wav")
    status_box.success(f"Whistle: {whistle_type} ({whistle_conf*100:.1f}%)")

    # --- Record Note ---
    status_box.info("Speak your note now!")
    processor.frames.clear()
    note_frames = []
    start = time.time()
    while time.time() - start < NOTE_DURATION:
        if processor.frames:
            note_frames.extend(processor.frames)
            processor.frames.clear()
        progress.progress(int(((time.time() - start) / NOTE_DURATION) * 100))
        time.sleep(0.05)
    save_audio(note_frames, "voice_note.wav")
    transcription = transcribe_audio("voice_note.wav")
    st.success(f"Whistle: {whistle_type}")
    st.write(f"**Transcript:** {transcription}")

    st.session_state.events.append({
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "whistle": f"{whistle_type} ({whistle_conf*100:.1f}%)",
        "note": transcription
    })
    print("[DEBUG] Event logged successfully")

# --- Event Log ---
st.write("### Event Log")
for e in st.session_state.events:
    st.write(f"- **{e['time']}**: Whistle: {e['whistle']} — Note: {e['note']}")

# --- RETRAINING SECTION ---
st.markdown("---")
st.header("Retrain Whistle Model")
single_samples = st.file_uploader("Upload Single Whistle Samples", type=["wav"], accept_multiple_files=True)
double_samples = st.file_uploader("Upload Double Whistle Samples", type=["wav"], accept_multiple_files=True)
if st.button("Retrain Model"):
    single_paths, double_paths = [], []
    for f in single_samples:
        path = os.path.join("data", f.name); open(path, "wb").write(f.read()); single_paths.append(path)
    for f in double_samples:
        path = os.path.join("data", f.name); open(path, "wb").write(f.read()); double_paths.append(path)
    model = retrain_model(single_paths, double_paths)
    if model:
        st.success("Model retrained successfully!")
    else:
        st.error("No training data provided.")