import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av, wave, os, time, numpy as np, matplotlib.pyplot as plt

# --- CONFIG ---
RATE = 44100
DURATION = 3
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

# --- AUDIO PROCESSOR (same as your code) ---
class AudioProcessor:
    def __init__(self):
        self.frames = []
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        self.frames.append(frame.to_ndarray().astype(np.int16).tobytes())
        return frame

# --- HELPER: Save WAV ---
def save_audio(frames, filename, rate=RATE):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))
    return filename

# --- UI ---
st.title("Audio Frame Capture Debugger")
progress = st.progress(0)
status = st.empty()

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
    st.info("Recording... Speak or whistle!")
    while time.time() - start < DURATION:
        if processor.frames:
            frames.extend(processor.frames)
            processor.frames.clear()
        if int(time.time() - start) % 1 == 0:
            print(f"[DEBUG] Frames so far: {len(frames)}")
        progress.progress(int(((time.time() - start) / DURATION) * 100))
        time.sleep(0.05)

    # Save file
    filename = "data/test_capture.wav"
    save_audio(frames, filename)
    st.success(f"Recording saved: {filename}")
    st.audio(filename, format="audio/wav")

    # Show stats
    file_size = os.path.getsize(filename)
    st.write(f"**Frames captured:** {len(frames)}")
    st.write(f"**File size:** {file_size} bytes")

    # Optional waveform visualization
    import soundfile as sf
    audio_data, sr = sf.read(filename)
    fig, ax = plt.subplots()
    ax.plot(audio_data)
    ax.set_title("Captured Audio Waveform")
    st.pyplot(fig)
