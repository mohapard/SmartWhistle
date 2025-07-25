import streamlit as st
import pyaudio, wave, openai, librosa, pickle, os, shutil
import numpy as np
from sklearn.linear_model import LogisticRegression
from datetime import datetime
import time
import matplotlib.pyplot as plt
import librosa.display

# --- CONFIG ---
openai.api_key = "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"
CHUNK, FORMAT, CHANNELS, RATE = 1024, pyaudio.paInt16, 1, 44100
MODEL_FILE = "whistle_mel_model.pkl"
DATASET_DIR = "whistle_dataset"
WHISTLE_DURATION = 3  # Force all whistle samples to 3 seconds

# --- SESSION STATE ---
if "events" not in st.session_state: st.session_state.events = []
if "last_transcription" not in st.session_state: st.session_state.last_transcription = ""
if "labels" not in st.session_state: st.session_state.labels = {"single": [], "double": []}

os.makedirs(DATASET_DIR, exist_ok=True)

# --- FIXED DURATION RECORDING (for whistles) ---
def record_fixed_audio(filename="sample.wav", record_seconds=WHISTLE_DURATION):
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames, start_time = [], time.time()
    max_chunks = int(RATE / CHUNK * record_seconds)
    progress = st.progress(0)
    timer_placeholder = st.empty()

    for i in range(max_chunks):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
        elapsed = time.time() - start_time
        progress.progress(min(1.0, elapsed / record_seconds))
        timer_placeholder.text(f"Recording whistle... {elapsed:.1f}s elapsed")

    stream.stop_stream(); stream.close(); p.terminate()
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS); wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE); wf.writeframes(b''.join(frames))
    timer_placeholder.text(f"Whistle recording finished ({elapsed:.1f}s).")
    return filename

# --- AUTO-STOP RECORDING (for voice notes) ---
def record_audio_autostop(filename="sample.wav", record_seconds=10, silence_threshold=500, min_duration=1,silence_stop=3):
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames, silent_chunks, start_time = [], 0, time.time()
    max_chunks = int(RATE / CHUNK * record_seconds)
    progress = st.progress(0)
    timer_placeholder = st.empty()

    for i in range(max_chunks):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
        audio_data = np.frombuffer(data, dtype=np.int16)
        if np.max(np.abs(audio_data)) < silence_threshold:
            silent_chunks += 1
        else:
            silent_chunks = 0
        elapsed = time.time() - start_time
        progress.progress(min(1.0, elapsed / record_seconds))
        timer_placeholder.text(f"Recording voice note... {elapsed:.1f}s elapsed")
        if silent_chunks > int(RATE / CHUNK * silence_stop) and elapsed > min_duration:
            break

    stream.stop_stream(); stream.close(); p.terminate()
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS); wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE); wf.writeframes(b''.join(frames))
    timer_placeholder.text(f"Voice note recording finished ({elapsed:.1f}s).")
    return filename

# --- MEL SPECTROGRAM EXTRACTION (Force 3s) ---
def audio_to_mel(filename, n_mels=40, fixed_duration=WHISTLE_DURATION, sr_target=22050):
    y, sr = librosa.load(filename, sr=sr_target)
    target_len = int(fixed_duration * sr)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.flatten()

# --- MODEL TRAINING ---
def train_classifier(single_files, double_files):
    X, y = [], []
    for f in single_files: X.append(audio_to_mel(f)); y.append(0)
    for f in double_files: X.append(audio_to_mel(f)); y.append(1)
    model = LogisticRegression(max_iter=2000)
    model.fit(X, y)
    with open(MODEL_FILE, "wb") as f: pickle.dump(model, f)
    return model

def load_model():
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "rb") as f: return pickle.load(f)
    return None

# --- PREDICTION ---
def predict_whistle(filename, model):
    features = audio_to_mel(filename)
    pred = model.predict([features])[0]
    prob = model.predict_proba([features])[0][pred]
    return ("Double Whistle" if pred == 1 else "Single Whistle"), prob

# --- TRANSCRIPTION & LOGGING ---
def transcribe_audio(filename):
    with open(filename, "rb") as audio_file:
        result = openai.audio.transcriptions.create(model="gpt-4o-mini-transcribe", file=audio_file)
    return result.text

def log_event(event_type, transcription):
    event = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
             "event_type": event_type, "description": transcription}
    st.session_state.events.append(event)
    st.session_state.last_transcription = transcription

# --- VISUALIZATION ---
def plot_mel(filename):
    y, sr = librosa.load(filename, sr=None)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    plt.figure(figsize=(6,3))
    librosa.display.specshow(log_mel, sr=sr, x_axis='time', y_axis='mel')
    plt.title("Mel Spectrogram")
    plt.colorbar(format='%+2.0f dB')
    st.pyplot(plt)

# --- UI ---
st.title("Smart Referee Whistle – Whistle + Voice Note")

mode = st.radio("Mode", ["Calibration (Data Collection)", "Detection", "Export Dataset"])

# --- CALIBRATION MODE ---
if mode == "Calibration (Data Collection)":
    st.write("Record samples for **Single** and **Double** whistles. Minimum 3 each for training.")
    label = st.selectbox("Label for next sample:", ["single", "double"])
    if st.button("Record Sample"):
        filename = f"{DATASET_DIR}/{label}_{len(st.session_state.labels[label]) + 1}.wav"
        record_fixed_audio(filename)
        st.session_state.labels[label].append(filename)
        st.success(f"{label.capitalize()} whistle sample recorded and saved as {filename}")
        st.subheader("Mel Spectrogram of Sample")
        plot_mel(filename)
    if st.button("Train & Save Model"):
        singles = st.session_state.labels["single"]
        doubles = st.session_state.labels["double"]
        if len(singles) >= 3 and len(doubles) >= 3:
            model = train_classifier(singles, doubles)
            st.success("Model trained & saved!")
        else:
            st.error("Need at least 3 samples each for single and double whistles.")

# --- DETECTION MODE ---
elif mode == "Detection":
    model = load_model()
    if model:
        if st.button("Record Whistle + Voice Note"):
            # Step 1: Record whistle
            st.info("Recording whistle...")
            whistle_file = record_fixed_audio("whistle_clip.wav")
            whistle_type, confidence = predict_whistle(whistle_file, model)
            st.success(f"{whistle_type} detected! (Confidence: {confidence:.1%})")
            #st.subheader("Mel Spectrogram of Whistle")
            #plot_mel(whistle_file)
            
            # Step 2: Record voice note (auto-stop)
            st.info("Now recording voice note (auto-stop when silent)...")
            voice_file = record_audio_autostop("voice_note.wav", record_seconds=10, silence_threshold=600, min_duration=1.5,silence_stop=3)
            
            # Step 3: Transcribe
            transcription = transcribe_audio(voice_file)
            #st.success("Voice note transcribed!")
            
            # Step 4: Log
            log_event(whistle_type, transcription)
            st.success("Event logged!")
    else:
        st.error("No trained model found. Calibrate first!")

    #if st.session_state.last_transcription:
    #    st.subheader("Last Transcribed Note")
    #    st.write(st.session_state.last_transcription)

    st.subheader("Event Log")
    for e in reversed(st.session_state.events):
        st.markdown(f"**[{e['timestamp']}] {e['event_type']}** – {e['description']}")

# --- EXPORT DATASET ---
elif mode == "Export Dataset":
    shutil.make_archive("whistle_dataset_export", 'zip', DATASET_DIR)
    with open("whistle_dataset_export.zip", "rb") as f:
        st.download_button("Download Labeled Dataset (.zip)", f, file_name="whistle_dataset.zip")
