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
RATE = 44100
os.makedirs(DATASET_DIR, exist_ok=True)

# --- SESSION STATE ---
if "events" not in st.session_state: st.session_state.events = []
if "labels" not in st.session_state: st.session_state.labels = {"single": [], "double": []}
if "recording_state" not in st.session_state: st.session_state.recording_state = "idle"
if "status" not in st.session_state: st.session_state.status = "Idle"
if "whistle_start" not in st.session_state: st.session_state.whistle_start = None
if "note_start" not in st.session_state: st.session_state.note_start = None

# --- AUDIO HELPERS ---
def save_audio(frames, filename="sample.wav", rate=RATE):
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

# --- Audio Processor: only collects frames ---
class AudioProcessor:
    def __init__(self):
        self.frames = []
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        pcm = frame.to_ndarray().tobytes()
        self.frames.append(pcm)
        return frame

# --- UI ---
st.title("Smart Whistle Logger (Debug Mode)")

status_box = st.empty()
status_box.info(st.session_state.status)
st.write(f"Current State: {st.session_state.recording_state}")

webrtc_ctx = webrtc_streamer(
    key="recorder",
    mode=WebRtcMode.SENDRECV,
    audio_receiver_size=256,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True,
    audio_processor_factory=AudioProcessor,
)

# --- DEBUGGING ---
def debug_log(msg):
    print(msg)
    st.write(f"DEBUG: {msg}")

# --- Start Recording ---
if webrtc_ctx.audio_receiver and st.button("Log Event"):
    processor = webrtc_ctx.audio_processor
    if processor:
        processor.frames.clear()
        st.session_state.recording_state = "whistle"
        st.session_state.whistle_start = time.time()
        st.session_state.status = "Recording whistle..."
        debug_log("Started whistle recording")

# --- State Machine in Main Script ---
processor = webrtc_ctx.audio_processor
if processor:
    now = time.time()

    # Whistle phase -> switch after WHISTLE_DURATION
    if st.session_state.recording_state == "whistle":
        elapsed = now - st.session_state.whistle_start
        debug_log(f"Whistle recording... {elapsed:.2f}s")
        if elapsed >= WHISTLE_DURATION:
            st.session_state.whistle_frames = processor.frames.copy()
            processor.frames.clear()
            st.session_state.recording_state = "note"
            st.session_state.note_start = now
            st.session_state.status = "Recording voice note..."
            debug_log("Whistle phase complete. Switched to note recording.")

    # Note phase -> stop after silence or VOICE_MAX
    elif st.session_state.recording_state == "note":
        elapsed = now - st.session_state.note_start
        debug_log(f"Note recording... {elapsed:.2f}s")
        if elapsed >= VOICE_MAX:
            st.session_state.note_frames = processor.frames.copy()
            st.session_state.recording_state = "done"
            st.session_state.status = "Processing..."
            debug_log("Max note duration reached. Finishing recording.")

        # Simple silence detection (last frame)
        if processor.frames:
            audio_data = np.frombuffer(processor.frames[-1], dtype=np.int16)
            if np.max(np.abs(audio_data)) < SILENCE_THRESHOLD:
                st.session_state.silent_chunks = st.session_state.get("silent_chunks", 0) + 1
            else:
                st.session_state.silent_chunks = 0
            if st.session_state.silent_chunks > int(RATE / 1024 * SILENCE_STOP):
                st.session_state.note_frames = processor.frames.copy()
                st.session_state.recording_state = "done"
                st.session_state.status = "Processing..."
                debug_log("Silence detected. Finishing recording.")

# --- When done: process results ---
if st.session_state.recording_state == "done":
    whistle_file, voice_file = "whistle.wav", "voice_note.wav"
    save_audio(st.session_state.whistle_frames, whistle_file)
    save_audio(st.session_state.note_frames, voice_file)
    whistle_type = predict_whistle(whistle_file)
    transcription = transcribe_audio(voice_file)
    debug_log(f"Whistle classified: {whistle_type}")
    debug_log(f"Transcription: {transcription}")

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
    st.session_state.status = "Idle"
    debug_log("Event logged and state reset to idle.")

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
