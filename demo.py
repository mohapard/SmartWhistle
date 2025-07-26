import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av, wave, os, pickle, time
import numpy as np
import openai, librosa
from sklearn.linear_model import LogisticRegression
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

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

# --- SESSION STATE INIT ---
defaults = {
    "recording_state": "idle",
    "whistle_frames": [],
    "note_frames": [],
    "whistle_start": None,
    "note_start": None,
    "events": [],
    "whistle_type": None
}
for k, v in defaults.items():
    if k not in st.session_state: st.session_state[k] = v

# --- HELPERS ---
def save_audio(frames, filename, rate=RATE):
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

def load_model():
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "rb") as f: return pickle.load(f)
    return None

def predict_whistle(filename):
    model = load_model()
    if model is None: return "Unknown Whistle"
    features = extract_mel(filename).reshape(1, -1)
    return "Single" if model.predict(features)[0] == 0 else "Double"

def transcribe_audio(filename):
    with open(filename, "rb") as f:
        transcript = openai.audio.transcriptions.create(model="gpt-4o-transcribe", file=f)
    return transcript.text

# --- AUDIO PROCESSOR ---
class AudioProcessor:
    def __init__(self):
        self.frames = []
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        pcm = frame.to_ndarray().astype(np.int16).tobytes()
        self.frames.append(pcm)
        return frame

# --- UI ---
st.title("Smart Whistle Logger")
st_autorefresh(interval=500, key="refresh")
status_box = st.empty()
progress = st.progress(0)

# --- WEBRTC ---
webrtc_ctx = webrtc_streamer(
    key="recorder",
    mode=WebRtcMode.SENDONLY,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True,
    audio_processor_factory=AudioProcessor,
    sendback_audio=False
)

# --- FRAME PULLER (main thread pulls from processor) ---
if webrtc_ctx.audio_processor:
    buffered = webrtc_ctx.audio_processor.frames
    if buffered:
        if st.session_state.recording_state == "whistle":
            st.session_state.whistle_frames.extend(buffered)
        elif st.session_state.recording_state == "note":
            st.session_state.note_frames.extend(buffered)
        webrtc_ctx.audio_processor.frames = []  # clear buffer

# --- STATE MACHINE ---
now = time.time()

# Start button
if webrtc_ctx.audio_receiver and st.button("Log Event"):
    st.session_state.recording_state = "whistle"
    st.session_state.whistle_start = now
    st.session_state.whistle_frames = []
    st.session_state.note_frames = []
    st.session_state.whistle_type = None
    print("[DEBUG] Started whistle recording")

# WHISTLE PHASE
if st.session_state.recording_state == "whistle":
    elapsed = now - st.session_state.whistle_start
    progress.progress(min(int((elapsed / WHISTLE_DURATION) * 100), 100))
    status_box.info(f"Blow your whistle... ({elapsed:.1f}/{WHISTLE_DURATION}s)")
    if elapsed >= WHISTLE_DURATION:
        save_audio(st.session_state.whistle_frames, "whistle.wav")
        whistle_type = predict_whistle("whistle.wav")
        st.session_state.whistle_type = whistle_type
        st.session_state.recording_state = "note"
        st.session_state.note_start = now
        print(f"[DEBUG] Whistle captured: {whistle_type}")

# NOTE PHASE
elif st.session_state.recording_state == "note":
    elapsed = now - st.session_state.note_start
    progress.progress(min(int((elapsed / NOTE_DURATION) * 100), 100))
    status_box.info(f"Recording voice note... ({elapsed:.1f}/{NOTE_DURATION}s)")
    if elapsed >= NOTE_DURATION:
        save_audio(st.session_state.note_frames, "voice_note.wav")
        transcription = transcribe_audio("voice_note.wav")
        whistle_type = st.session_state.whistle_type or "Unknown"
        st.session_state.events.append({
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "whistle": whistle_type,
            "note": transcription
        })
        st.success(f"Whistle: {whistle_type}")
        st.write(f"**Transcript:** {transcription}")
        st.session_state.recording_state = "idle"
        progress.progress(0)
        print(f"[DEBUG] Note recorded & transcribed: {transcription}")

# IDLE
elif st.session_state.recording_state == "idle":
    status_box.info("Idle - Click 'Log Event' to start")
    progress.progress(0)

# --- Event Log ---
st.write("### Event Log")
for e in st.session_state.events:
    st.write(f"- **{e['time']}**: Whistle: {e['whistle']} — Note: {e['note']}")
