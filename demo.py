import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av, wave, os, time, numpy as np, soundfile as sf, librosa, matplotlib.pyplot as plt

DURATION = 3
os.makedirs("data", exist_ok=True)
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# --- Convert planar float32 → interleaved int16 PCM ---
def planar_to_interleaved_pcm16(arr):
    if arr.ndim == 2:  # Planar: (channels, samples)
        arr = arr.T.flatten()  # Convert to interleaved
    arr = np.clip(arr, -1.0, 1.0)  # Keep within valid range
    return (arr * 32767).astype(np.int16)

# --- AUDIO PROCESSOR (with diagnostics) ---
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
        # Capture diagnostics
        self.last_format = frame.format.name
        self.last_layout = frame.layout.name
        self.last_samples = frame.samples
        self.sample_rate = frame.sample_rate
        try:
            self.channels = frame.layout.nb_channels
        except Exception:
            self.channels = 1  # Fallback to mono if unavailable

        # Log to server console
        print(f"[DEBUG] Frame: format={self.last_format}, layout={self.last_layout}, "
              f"rate={self.sample_rate}, channels={self.channels}, samples={self.last_samples}")

        # Convert to interleaved PCM16
        arr = frame.to_ndarray()
        self.last_shape = arr.shape
        pcm16 = planar_to_interleaved_pcm16(arr)
        self.frames.append(pcm16.tobytes())
        return frame

# --- Save WAV ---
def save_audio(frames, filename, rate=None, channels=None):
    if not frames:
        print("[DEBUG] No frames to save.")
        return None
    try:
        rate = int(rate) if rate else 48000
    except Exception:
        rate = 48000
    try:
        channels = int(channels) if isinstance(channels, (int, float)) else 1
    except Exception:
        channels = 1
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))
    return filename, rate, channels

# --- Plot waveform ---
def plot_waveform(filename, title):
    data, sr = sf.read(filename)
    fig, ax = plt.subplots()
    ax.plot(data)
    ax.set_title(f"{title} ({sr}Hz)")
    st.pyplot(fig)

# --- UI ---
st.title("Golden Diagnostic Audio Tester")
progress = st.progress(0)

webrtc_ctx = webrtc_streamer(
    key="diagnostic",
    mode=WebRtcMode.SENDONLY,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True,
    audio_processor_factory=AudioProcessor,
    sendback_audio=False
)

if webrtc_ctx.audio_processor and st.button("Record & Diagnose"):
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

    # --- Save file ---
    raw_file = "data/diagnostic.wav"
    result = save_audio(frames, raw_file, processor.sample_rate, processor.channels)
    if not result:
        st.error("No audio captured. Check mic permissions or try again.")
    else:
        raw_file, rate, channels = result
        st.success(f"Saved: {raw_file} ({rate}Hz, {channels}ch)")
        st.write(f"**Frame format:** {processor.last_format}")
        st.write(f"**Layout:** {processor.last_layout}")
        st.write(f"**Samples per frame:** {processor.last_samples}")
        st.write(f"**NDArray shape:** {processor.last_shape}")
        st.audio(raw_file, format="audio/wav")
        plot_waveform(raw_file, "Captured Audio")

        # Debug info
        st.write(f"**Frames captured:** {len(frames)}")
        st.write(f"**File size:** {os.path.getsize(raw_file)} bytes")