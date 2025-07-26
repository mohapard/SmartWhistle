import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av, wave, os, time, numpy as np, matplotlib.pyplot as plt

DURATION = 3
os.makedirs("data", exist_ok=True)
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

class AudioProcessor:
    def __init__(self):
        self.frames = []
        self.sample_rate = None
        self.channels = None
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        arr = frame.to_ndarray().astype(np.int16)
        self.frames.append(arr.tobytes())
        # Save actual properties from first frame
        if self.sample_rate is None:
            self.sample_rate = frame.sample_rate
            self.channels = frame.layout.channels
        return frame

def save_audio(frames, filename, rate, channels):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))
    return filename

st.title("Audio Capture Debugger (Correct Rate)")
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
    st.info("Recording...")
    while time.time() - start < DURATION:
        if processor.frames:
            frames.extend(processor.frames)
            processor.frames.clear()
        time.sleep(0.05)

    rate = processor.sample_rate or 48000
    channels = processor.channels or 1
    filename = "data/test_correct.wav"
    save_audio(frames, filename, rate, channels)
    st.success(f"Recording saved: {filename} ({rate}Hz, {channels}ch)")
    st.audio(filename, format="audio/wav")

    # Plot waveform
    import soundfile as sf
    audio_data, sr = sf.read(filename)
    fig, ax = plt.subplots()
    ax.plot(audio_data)
    ax.set_title(f"Waveform ({sr}Hz)")
    st.pyplot(fig)
