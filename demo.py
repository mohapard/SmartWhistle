import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av, wave, os, time, numpy as np, soundfile as sf, matplotlib.pyplot as plt

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

# --- SAVE FUNCTION ---
def save_audio(frames, filename, rate=None, channels=None):
    if not frames:
        print("[DEBUG] No frames to save.")
        return None
    # Force safe defaults if None or 0
    rate = rate if rate and rate > 0 else 48000
    channels = channels if channels and channels > 0 else 1
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))
    return filename

# --- UI ---
st.title("Audio Capture Tester (Correct Rate)")
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

    # Save file using detected rate/channels
    #rate = processor.sample_rate or 48000
    #channels = processor.channels or 1
    filename = "data/test_fixed.wav"
    rate = processor.sample_rate
    channels = processor.channels
    save_audio(frames, filename, rate, channels)
    st.success(f"Recording saved: {filename} ({rate}Hz, {channels}ch)")
    st.audio(filename, format="audio/wav")

    # Plot waveform
    data, sr = sf.read(filename)
    fig, ax = plt.subplots()
    ax.plot(data)
    ax.set_title(f"Captured Audio Waveform ({sr}Hz)")
    st.pyplot(fig)

    # Show debug info
    st.write(f"**Frames captured:** {len(frames)}")
    st.write(f"**File size:** {os.path.getsize(filename)} bytes")
