import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av, wave, os, pickle, time, shutil
import numpy as np
import openai, librosa
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
if "recording_state" not in st.session_state: st.session_state.recording_state = "idle"
if "status" not in st.session_state: st.session_state.status = "Idle"

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
    if model is None: return "Unknown Whistle"
    features = extract_mel(filename).reshape(1, -1)
    return "Single" if model.predict(features)[0] == 0 else "Double"

# --- Transcription ---
def transcribe_audio(filename):
    with open(filename, "rb") as f:
        transcript = openai.audio.transcriptions.create(model="gpt-4o-transcribe", file=f)
    return transcript.text

# --- Audio Processor with State Machine ---
class AudioProcessor:
    def __init__(self):
        self.frames = []
        self.whistle_frames = []
        self.note_frames = []
        self.state = "idle"
        self.start_time = None
        self.silent_chunks = 0

    def start_whistle(self):
        self.frames.clear()
        self.state = "whistle"
        self.start_time = time.time()
        st.session_state.status = "Recording whistle..."

    def start_note(self):
        self.frames.clear()
        self.state = "note"
        self.start_time = time.time()
        self.silent_chunks = 0
        st.session_state.status = "Recording voice note (speak now)..."

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        pcm = frame.to_ndarray().tobytes()

        if self.state == "whistle":
            self.frames.append(pcm)
            if time.time() - self.start_time >= WHISTLE_DURATION:
                self.whistle_frames = self.frames.copy()
                self.start_note()

        elif self.state == "note":
            self.frames.append(pcm)
            audio_data = np.frombuffer(pcm, dtype=np.int16)
            if np.max(np.abs(audio_data)) < SILENCE_THRESHOLD:
                self.silent_chunks += 1
            else:
                self.silent_chunks = 0
            if (self.silent_chunks > int(44100 / 1024 * SILENCE_STOP)) or (time.time() - self.start_time >= VOICE_MAX):
                self.note_frames = self.frames.copy()
                self.state = "done"
                st.session_state.status = "Processing..."

        return frame

# --- UI ---
st.title("Smart Whistle Logger")
st.write("One click: Record whistle → voice note → transcribe → log.")

status_box = st.empty()
status_box.info(st.session_state.status)

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
        processor.start_whistle()
        st.session_state.recording_state = "recording"

if webrtc_ctx.audio_processor and webrtc_ctx.audio_processor.state == "note":
    if st.button("STOP Recording"):
        webrtc_ctx.audio_processor.state = "done"
        st.session_state.status = "Processing..."

# --- When done: process results ---
if webrtc_ctx.audio_processor and webrtc_ctx.audio_processor.state == "done" and st.session_state.recording_state == "recording":
    processor = webrtc_ctx.audio_processor
    whistle_file, voice_file = "whistle.wav", "voice_note.wav"
    save_audio(processor.whistle_frames, whistle_file)
    save_audio(processor.note_frames, voice_file)
    whistle_type = predict_whistle(whistle_file)
    transcription = transcribe_audio(voice_file)

    # Log event
    event = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "whistle": whistle_type,
        "note": transcription
    }
    st.session_state.events.append(event)

    st.success(f"Whistle: {whistle_type}")
    st.write(f"**Transcript:** {transcription}")
    st.session_state.recording_state = "idle"
    webrtc_ctx.audio_processor.state = "idle"
    st.session_state.status = "Idle"

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
