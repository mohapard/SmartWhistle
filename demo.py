import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av, wave, os, pickle, time, shutil
import numpy as np
import openai, librosa
from sklearn.linear_model import LogisticRegression
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

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

# --- SESSION STATE ---
if "events" not in st.session_state: st.session_state.events = []
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

# --- Audio Processor ---
class AudioProcessor:
    def __init__(self):
        self.frames = []
        self.count = 0
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        self.count += 1
        self.frames.append(frame.to_ndarray().astype(np.int16).tobytes())
        return frame

# --- UI ---
st.title("Smart Whistle Logger")
st_autorefresh(interval=500, key="refresh")

status_box = st.empty()
progress = st.progress(0)

# WebRTC
webrtc_ctx = webrtc_streamer(
    key="recorder",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    audio_receiver_size=256,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True,
    audio_processor_factory=AudioProcessor,
)

# --- START RECORDING ---
if webrtc_ctx.audio_receiver and st.button("Log Event"):
    processor = webrtc_ctx.audio_processor
    if processor:
        processor.frames.clear()
        st.session_state.recording_state = "whistle"
        st.session_state.whistle_start = time.time()
        st.session_state.status = "Blow your whistle now!"
        print("[DEBUG] Started whistle recording")

processor = webrtc_ctx.audio_processor
if processor:
    now = time.time()

    # --- WHISTLE PHASE ---
    if st.session_state.recording_state == "whistle":
        elapsed = now - st.session_state.whistle_start
        pct = min(int((elapsed / WHISTLE_DURATION) * 100), 100)
        progress.progress(pct)
        status_box.info(f"Blow your whistle... ({elapsed:.1f}/{WHISTLE_DURATION}s)")
        print(f"[DEBUG] Whistle recording... {elapsed:.2f}s ({pct}%)")
        if elapsed >= WHISTLE_DURATION:
            st.session_state.whistle_frames = processor.frames.copy()
            save_audio(st.session_state.whistle_frames, "whistle.wav")
            whistle_type = predict_whistle("whistle.wav")
            print(f"[DEBUG] Whistle classified: {whistle_type}")
            st.session_state.whistle_type = whistle_type
            processor.frames.clear()
            st.session_state.recording_state = "note"
            st.session_state.note_start = now
            st.session_state.status = f"Whistle detected: {whistle_type}. Speak your note now!"
            progress.progress(0)
            print("[DEBUG] Whistle phase complete. Switched to note recording.")

    # --- NOTE PHASE ---
    elif st.session_state.recording_state == "note":
        elapsed = now - st.session_state.note_start
        pct = min(int((elapsed / VOICE_MAX) * 100), 100)
        progress.progress(pct)
        status_box.info(f"Recording voice note... ({elapsed:.1f}/{VOICE_MAX}s)")
        print(f"[DEBUG] Note recording... {elapsed:.2f}s ({pct}%)")
        if elapsed >= VOICE_MAX:
            st.session_state.note_frames = processor.frames.copy()
            st.session_state.recording_state = "done"
            st.session_state.status = "Processing..."
            print("[DEBUG] Max note duration reached.")
        # Silence stop
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
                print("[DEBUG] Silence detected. Finishing note.")

    # --- PROCESS RESULTS ---
    elif st.session_state.recording_state == "done":
        status_box.info("Processing...")
        save_audio(st.session_state.note_frames, "voice_note.wav")
        transcription = transcribe_audio("voice_note.wav")
        whistle_type = st.session_state.get("whistle_type", "Unknown")
        print(f"[DEBUG] Transcription: {transcription}")

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
        progress.progress(0)
        print("[DEBUG] Event logged and reset.")

    # --- IDLE PHASE ---
    else:
        status_box.info("Idle - Click 'Log Event' to start")
        progress.progress(0)
        print("[DEBUG] Idle state.")

# --- Event Log ---
st.write("### Event Log")
for e in st.session_state.events:
    st.write(f"- **{e['time']}**: Whistle: {e['whistle']} — Note: {e['note']}")
