import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av, wave, time, shutil, os, pickle
import numpy as np
import openai
import librosa, librosa.display
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from datetime import datetime

# --- CONFIG ---
openai.api_key = st.secrets["openai_key"]
MODEL_FILE = "whistle_mel_model.pkl"
DATASET_DIR = "whistle_dataset"
WHISTLE_DURATION = 3
VOICE_MAX = 10
SILENCE_STOP = 3
SILENCE_THRESHOLD = 500
os.makedirs(DATASET_DIR, exist_ok=True)

# --- SESSION STATE ---
if "events" not in st.session_state: st.session_state.events = []
if "labels" not in st.session_state: st.session_state.labels = {"single": [], "double": []}

# --- AUDIO HELPERS ---
def save_audio(frames, filename="sample.wav", rate=44100):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))

def extract_mel(filename):
    y, sr = librosa.load(filename, sr=22050)
    y = librosa.util.fix_length(y, size=sr * WHISTLE_DURATION)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
    return librosa.power_to_db(mel, ref=np.max).flatten()

# --- MODEL ---
def train_classifier(single_whistles, double_whistles):
    X, y = [], []
    for f in single_whistles:
        X.append(extract_mel(f)); y.append(0)
    for f in double_whistles:
        X.append(extract_mel(f)); y.append(1)
    model = LogisticRegression(max_iter=500)
    model.fit(X, y)
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    return model

def load_model():
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "rb") as f: return pickle.load(f)
    return None

def predict_whistle(filename):
    model = load_model()
    if model is None: return "Untrained"
    features = extract_mel(filename).reshape(1, -1)
    return "Single" if model.predict(features)[0] == 0 else "Double"

# --- Transcription ---
def transcribe_audio(filename):
    with open(filename, "rb") as f:
        transcript = openai.audio.transcriptions.create(model="gpt-4o-transcribe", file=f)
    return transcript.text

# --- Audio Processor for WebRTC ---
class AudioProcessor:
    def __init__(self): self.frames = []
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        pcm = frame.to_ndarray().tobytes()
        self.frames.append(pcm)
        return frame

# --- SILENCE DETECTION ---
def capture_until_silence(frames, start_idx=0, min_len=1, max_len=VOICE_MAX, silence_stop=SILENCE_STOP, silence_thresh=SILENCE_THRESHOLD, rate=44100):
    captured = []
    silent_chunks, start_time = 0, time.time()
    chunk_size = 1024
    max_chunks = int(rate / chunk_size * max_len)
    while True:
        if len(frames) > start_idx:
            frame = frames[start_idx]
            captured.append(frame)
            start_idx += 1
            audio_data = np.frombuffer(frame, dtype=np.int16)
            if np.max(np.abs(audio_data)) < silence_thresh:
                silent_chunks += 1
            else:
                silent_chunks = 0
        if len(captured) >= max_chunks: break
        if silent_chunks > int(rate / chunk_size * silence_stop) and (time.time() - start_time) > min_len:
            break
        time.sleep(0.05)
    return captured

# --- UI ---
st.title("Smart Whistle Logger")

st.write("Click below to log a whistle event (detect whistle → record note → transcribe → log):")
webrtc_ctx = webrtc_streamer(
    key="recorder",
    mode=WebRtcMode.SENDRECV,
    audio_receiver_size=256,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True,
    audio_processor_factory=AudioProcessor,
)

if webrtc_ctx.audio_receiver and st.button("Log Event"):
    processor = webrtc_ctx.audio_processor
    if processor:
        # Reset buffer
        processor.frames.clear()

        # --- Step 1: Whistle ---
        st.info("Recording whistle...")
        start = time.time()
        while time.time() - start < WHISTLE_DURATION:
            time.sleep(0.05)
        whistle_frames = processor.frames.copy()
        save_audio(whistle_frames, "whistle.wav")
        whistle_type = predict_whistle("whistle.wav")
        st.success(f"Whistle detected: {whistle_type}")

        # --- Step 2: Voice Note ---
        st.info("Recording voice note...")
        processor.frames.clear()
        voice_frames = capture_until_silence(processor.frames)
        save_audio(voice_frames, "voice_note.wav")
        st.success("Voice note captured.")

        # --- Step 3: Transcribe ---
        st.info("Transcribing...")
        transcription = transcribe_audio("voice_note.wav")
        st.write(f"**Transcript:** {transcription}")

        # --- Step 4: Log Event ---
        event = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "whistle": whistle_type,
            "note": transcription
        }
        st.session_state.events.append(event)
        st.success("Event logged!")

# --- Event Log ---
st.write("### Event Log")
for e in st.session_state.events:
    st.write(f"- **{e['time']}**: Whistle: {e['whistle']} — Note: {e['note']}")

# --- Training Section ---
st.write("### Train Whistle Model")
label = st.selectbox("Label whistle sample as:", ["single", "double"])
if st.button("Save Whistle Sample"):
    if os.path.exists("whistle.wav"):
        dest = os.path.join(DATASET_DIR, f"{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav")
        shutil.copy("whistle.wav", dest)
        st.session_state.labels[label].append(dest)
        st.success(f"Saved {label} whistle sample.")
if st.button("Train Model"):
    singles, doubles = st.session_state.labels["single"], st.session_state.labels["double"]
    if singles and doubles:
        train_classifier(singles, doubles)
        st.success("Model trained.")
    else:
        st.warning("Need samples for both whistle types.")
