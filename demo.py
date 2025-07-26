import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av, wave, os, time, numpy as np, soundfile as sf, librosa, matplotlib.pyplot as plt

DURATION = 3
os.makedirs("data", exist_ok=True)
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# --- AUDIO PROCESSOR ---
class AudioProcessor:
    def __init__(self):
        self.frames = []
        self.sample_rate = None
        self.channels = None
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        arr = frame.to_ndarray().astype(np.int16)
        self.frames.append(arr.tobytes())
        if self.sample_rate is None:
            self.sample_rate = frame.sample_rate
        if self.channels is None:
            self.channels = frame.layout.channels
        return frame

# --- RAW SAVE FUNCTION ---
def save_audio(frames, filename, rate=None, channels=None):
    if not frames:
        print("[DEBUG] No frames to save.")
        return None
    # Force clean int values
    try:
        rate = int(rate) if rate else 48000
    except Exception:
        rate = 48000
    try:
        channels = int(channels) if channels else 1
    except Exception:
        channels = 1

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))
    return filename, rate, channels

# --- RESAMPLING TO 16k MONO ---
def resample_to_16k(input_file, output_file):
    data, sr = sf.read(input_file)
    if data.ndim > 1:
        data = np.mean(data, axis=1)  # Stereo → mono
    resampled = librosa.resample(data.astype(np.float32), orig_sr=sr, target_sr=16000)
    sf.write(output_file, resampled, 16000, subtype='PCM_16')
    return output_file

# --- WAVEFORM PLOT ---
def plot_waveform(filename, title):
    data, sr = sf.read(filename)
    fig, ax = plt.subplots()
    ax.plot(data)
    ax.set_title(f"{title} ({sr}Hz)")
    st.pyplot(fig)

# --- UI ---
st.title("Audio Capture Tester (Raw vs Resampled)")
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

if webrtc_ctx.audio_processor and st.button("Record Test Clip"):
    processor = webrtc_ctx.audio_processor
    processor.frames.clear()
    frames = []
    start = time.time()
    st.info("Recording 3 seconds...")
    while time.time() - start < DURATION:
        if processor.frames:
            frames.extend(processor.frames)
            processor.frames.clear()
        progress.progress(int(((time.time() - start) / DURATION) * 100))
        time.sleep(0.05)

    # --- Save raw ---
    raw_file = "data/test_raw.wav"
    raw_file, rate, channels = save_audio(frames, raw_file, processor.sample_rate, processor.channels)
    st.success(f"Raw saved: {raw_file} ({rate}Hz, {channels}ch)")
    st.audio(raw_file, format="audio/wav")
    plot_waveform(raw_file, "Raw WebRTC Audio")

    # --- Save resampled 16k mono ---
    clean_file = "data/test_resampled.wav"
    resample_to_16k(raw_file, clean_file)
    st.success(f"Resampled saved: {clean_file} (16kHz mono)")
    st.audio(clean_file, format="audio/wav")
    plot_waveform(clean_file, "Resampled Audio (16kHz Mono)")

    # Debug info
    st.write(f"**Frames captured:** {len(frames)}")
    st.write(f"**Raw file size:** {os.path.getsize(raw_file)} bytes")
    st.write(f"**Resampled file size:** {os.path.getsize(clean_file)} bytes")