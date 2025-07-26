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
        self.frames.append(frame.to_ndarray().astype(np.int16).tobytes())
        return frame

# --- UI ---
st.title("Smart Whistle Logger")
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
    whistle_type = predict_whistle("whistle.wav")
    status_box.success(f"Whistle detected: {whistle_type}")

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
        "whistle": whistle_type,
        "note": transcription
    })

# --- Event Log ---
st.write("### Event Log")
for e in st.session_state.events:
    st.write(f"- **{e['time']}**: Whistle: {e['whistle']} — Note: {e['note']}")
