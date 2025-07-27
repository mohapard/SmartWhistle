import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av, wave, os, time, pickle, numpy as np, soundfile as sf, librosa
from datetime import datetime
import openai
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle

# --- CONFIG ---
openai.api_key = st.secrets["openai_key"]
MODEL_FILE = "whistle_mel_model.pkl"
WHISTLE_DURATION = 3
NOTE_DURATION = 6
os.makedirs("data", exist_ok=True)
os.makedirs("data/retrain/single", exist_ok=True)
os.makedirs("data/retrain/double", exist_ok=True)
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# --- Audio Conversion ---
def to_pcm16(arr, fmt):
    if arr.ndim == 2:
        arr = arr.T.flatten()
    if fmt == "s16":
        return arr.astype(np.int16)
    else:
        arr = np.clip(arr, -1.0, 1.0)
        return (arr * 32767).astype(np.int16)

def save_wav(frames, filename, rate, channels):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(int(channels))
        wf.setsampwidth(2)
        wf.setframerate(int(rate))
        wf.writeframes(b''.join(frames))
    return filename

def resample_to_16k(input_file, output_file):
    data, sr = sf.read(input_file)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    resampled = librosa.resample(data.astype(np.float32), orig_sr=sr, target_sr=16000)
    sf.write(output_file, resampled, 16000, subtype='PCM_16')
    return output_file

# --- Feature Extraction & Model ---
def extract_mel(filename):
    y, sr = librosa.load(filename, sr=22050)
    y = librosa.util.fix_length(y, size=sr * WHISTLE_DURATION)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
    return librosa.power_to_db(mel, ref=np.max).flatten()

def load_model():
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "rb") as f:
            return pickle.load(f)
    return None

def predict_whistle(filename):
    model = load_model()
    if not model:
        return "Unknown Whistle"
    features = extract_mel(filename).reshape(1, -1)
    pred = model.predict(features)[0]
    return "Single" if pred == 0 else "Double"

def retrain_model():
    X, y = [], []
    for label, folder in enumerate(["single", "double"]):
        folder_path = os.path.join("data/retrain", folder)
        for f in os.listdir(folder_path):
            if f.endswith(".wav"):
                feat = extract_mel(os.path.join(folder_path, f))
                X.append(feat)
                y.append(label)
    if not X or len(set(y)) < 2:
        return False, "Need samples for both Single and Double whistles."
    X, y = shuffle(X, y, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    return True, "Model retrained successfully!"

# --- Whisper Transcription ---
def transcribe_audio(filename):
    with open(filename, "rb") as f:
        transcript = openai.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=f
        )
    return transcript.text

# --- AUDIO PROCESSOR ---
class AudioProcessor:
    def __init__(self):
        self.frames = []
        self.sample_rate = None
        self.channels = None
        self.format = None
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        self.format = frame.format.name
        self.sample_rate = frame.sample_rate
        try:
            self.channels = frame.layout.nb_channels
        except Exception:
            self.channels = 1
        arr = frame.to_ndarray()
        pcm16 = to_pcm16(arr, self.format)
        self.frames.append(pcm16.tobytes())
        return frame

# --- Helper: Sample counts ---
def get_sample_counts():
    single_count = len([f for f in os.listdir("data/retrain/single") if f.endswith(".wav")])
    double_count = len([f for f in os.listdir("data/retrain/double") if f.endswith(".wav")])
    return single_count, double_count

# --- UI ---
st.title("Smart Whistle Logger with Retraining")
status_box = st.empty()
progress = st.progress(0)
mode = st.radio("Select Mode", ["Logging", "Retraining"])

webrtc_ctx = webrtc_streamer(
    key="whistle_logger",
    mode=WebRtcMode.SENDONLY,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True,
    audio_processor_factory=AudioProcessor,
    sendback_audio=False
)

if "events" not in st.session_state:
    st.session_state.events = []

# --- Retraining Mode ---
if mode == "Retraining" and webrtc_ctx.audio_processor:
    single_count, double_count = get_sample_counts()
    st.markdown(f"**Current Samples:** Single: `{single_count}` | Double: `{double_count}`")
    label = st.selectbox("Whistle Type", ["Single", "Double"])
    if st.button("Record Sample"):
        processor = webrtc_ctx.audio_processor
        processor.frames.clear()
        frames = []
        status_box.info(f"Blow a {label} whistle!")
        start = time.time()
        while time.time() - start < WHISTLE_DURATION:
            if processor.frames:
                frames.extend(processor.frames)
                processor.frames.clear()
            progress.progress(int(((time.time() - start) / WHISTLE_DURATION) * 100))
            time.sleep(0.05)
        fname = f"data/retrain/{label.lower()}/{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        save_wav(frames, fname, processor.sample_rate or 48000, processor.channels or 2)
        status_box.success(f"Saved new {label} whistle sample: {fname}")

    if st.button("Retrain Model"):
        status_box.info("Retraining model...")
        success, msg = retrain_model()
        if success:
            status_box.success(msg)
        else:
            status_box.error(msg)

# --- Logging Mode ---
if mode == "Logging" and webrtc_ctx.audio_processor and st.button("Log Event"):
    processor = webrtc_ctx.audio_processor
    processor.frames.clear()
    frames = []

    # --- Step 1: Whistle ---
    status_box.info("Blow your whistle now!")
    start = time.time()
    while time.time() - start < WHISTLE_DURATION:
        if processor.frames:
            frames.extend(processor.frames)
            processor.frames.clear()
        progress.progress(int(((time.time() - start) / WHISTLE_DURATION) * 100))
        time.sleep(0.05)
    whistle_file = "data/whistle.wav"
    save_wav(frames, whistle_file, processor.sample_rate or 48000, processor.channels or 2)
    whistle_type = predict_whistle(whistle_file)
    status_box.success(f"Whistle detected: {whistle_type}")

    # --- Step 2: Voice Note ---
    status_box.info("Speak your note now!")
    frames = []
    processor.frames.clear()
    start = time.time()
    while time.time() - start < NOTE_DURATION:
        if processor.frames:
            frames.extend(processor.frames)
            processor.frames.clear()
        progress.progress(int(((time.time() - start) / NOTE_DURATION) * 100))
        time.sleep(0.05)
    note_file = "data/voice_note_raw.wav"
    save_wav(frames, note_file, processor.sample_rate or 48000, processor.channels or 2)
    whisper_file = "data/voice_note_16k.wav"
    resample_to_16k(note_file, whisper_file)

    # --- Step 3: Transcription ---
    status_box.info("Transcribing voice note...")
    transcription = transcribe_audio(whisper_file)
    status_box.success("Transcription complete!")

    # --- Save event ---
    st.session_state.events.append({
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "whistle": whistle_type,
        "note": transcription
    })
    st.success(f"Event logged! Whistle: {whistle_type}")
    st.write(f"**Transcript:** {transcription}")

# --- Event Log ---
st.write("### Event Log")
for e in st.session_state.events:
    st.write(f"- **{e['time']}**: Whistle: {e['whistle']} — Note: {e['note']}")
