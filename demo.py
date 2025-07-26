import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av, wave, os, time, numpy as np, soundfile as sf, librosa, matplotlib.pyplot as plt

DURATION = 3
os.makedirs("data", exist_ok=True)
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# --- Convert ndarray to interleaved PCM16 ---
def to_pcm16(arr, fmt):
    if arr.ndim == 2:  # (channels, samples)
        arr = arr.T.flatten()  # Interleave
    if fmt == "s16":
        # Already int16 PCM data
        return arr.astype(np.int16)
    else:
        # Float32/float64 → scale to PCM16
        arr = np.clip(arr, -1.0, 1.0)
        return (arr * 32767).astype(np.int16)

# --- AUDIO PROCESSOR ---
class AudioProcessor:
    def __init__(self):
        self.frames = []
        self.sample_rate = None
        self.channels = None
        self.last_shape = None
        self.last_format = None
        self.last_layout = None
        self.last_samples = None

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        self.last_format = frame.format.name
        self.last_layout = frame.layout.name
        self.last_samples = frame.samples
        self.sample_rate = frame.sample_rate
        try:
            self.channels = frame.layout.nb_channels
        except Exception:
            self.channels = 1

        # Extract as ndarray (always works)
        arr = frame.to_ndarray()
        self.last_shape = arr.shape
        pcm16 = to_pcm16(arr, self.last_format)
        self.frames.append(pcm16.tobytes())
        return frame

# --- Save WAV ---
def save_wav(frames, filename, rate, channels):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(int(channels))
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(int(rate))
        wf.writeframes(b''.join(frames))
    return filename

# --- Resample to 16k mono for Whisper ---
def resample_to_16k(input_file, output_file):
    data, sr = sf.read(input_file)
    if data.ndim > 1:
        data = np.mean(data, axis=1)  # Stereo → mono
    resampled = librosa.resample(data.astype(np.float32), orig_sr=sr, target_sr=16000)
    sf.write(output_file, resampled, 16000, subtype='PCM_16')
    return output_file

# --- Plot waveform ---
def plot_waveform(filename, title):
    data, sr = sf.read(filename)
    fig, ax = plt.subplots()
    ax.plot(data)
    ax.set_title(f"{title} ({sr}Hz)")
    st.pyplot(fig)

# --- UI ---
st.title("Golden Audio Recorder (Reliable Capture)")
progress = st.progress(0)

webrtc_ctx = webrtc_streamer(
    key="golden",
    mode=WebRtcMode.SENDONLY,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True,
    audio_processor_factory=AudioProcessor,
    sendback_audio=False
)

if webrtc_ctx.audio_processor and st.button("Record & Save Both"):
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

    if not frames:
        st.error("No audio captured. Check mic permissions or try again.")
    else:
        # --- Save RAW 48kHz stereo ---
        raw_file = "data/golden_raw.wav"
        save_wav(frames, raw_file, processor.sample_rate or 48000, processor.channels or 2)
        st.success(f"Raw saved: {raw_file} ({processor.sample_rate}Hz, {processor.channels}ch)")
        st.write(f"**Format:** {processor.last_format}")
        st.write(f"**Layout:** {processor.last_layout}")
        st.write(f"**Samples/frame:** {processor.last_samples}")
        st.write(f"**NDArray shape:** {processor.last_shape}")
        st.audio(raw_file, format="audio/wav")
        plot_waveform(raw_file, "Raw 48kHz Audio")

        # --- Save Whisper-ready 16kHz mono ---
        whisper_file = "data/golden_whisper.wav"
        resample_to_16k(raw_file, whisper_file)
        st.success(f"Whisper-ready saved: {whisper_file} (16kHz mono)")
        st.audio(whisper_file, format="audio/wav")
        plot_waveform(whisper_file, "Whisper 16kHz Mono")