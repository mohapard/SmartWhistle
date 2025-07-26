import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av, time, numpy as np, soundfile as sf, librosa, openai, matplotlib.pyplot as plt

openai.api_key = st.secrets["openai_key"]

INPUT_RATE = 48000  # WebRTC often uses 48kHz
TARGET_RATE = 16000
DURATION = 3
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# --- Correct Audio Saving ---
def save_audio(frames, filename, target_rate=TARGET_RATE, input_rate=INPUT_RATE):
    audio_bytes = b''.join(frames)
    if len(audio_bytes) == 0:
        print("[DEBUG] No audio data captured.")
        return None, None
    pcm = np.frombuffer(audio_bytes, dtype=np.float32)
    if pcm.ndim > 1:
        pcm = np.mean(pcm, axis=1)
    pcm_resampled = librosa.resample(pcm, orig_sr=input_rate, target_sr=target_rate)
    pcm_int16 = np.int16(np.clip(pcm_resampled, -1.0, 1.0) * 32767)
    sf.write(filename, pcm_int16, target_rate, subtype='PCM_16')
    print(f"[DEBUG] Saved {filename}, samples: {len(pcm_int16)}")
    return filename, pcm_resampled

class AudioProcessor:
    def __init__(self):
        self.frames = []
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        arr = frame.to_ndarray(format="flt")
        if arr.size > 0:
            print(f"[DEBUG] Frame captured: shape={arr.shape}, dtype={arr.dtype}")
        self.frames.append(arr.tobytes())
        return frame

st.title("Audio Capture Debug")
st.write("Record 3 seconds and inspect frame data")

webrtc_ctx = webrtc_streamer(
    key="recorder",
    mode=WebRtcMode.SENDRECV,  # <-- Force full streaming
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True,
    audio_processor_factory=AudioProcessor,
    sendback_audio=False
)

if webrtc_ctx.audio_processor and st.button("Record & Test"):
    st.write("Recording...")
    processor = webrtc_ctx.audio_processor
    processor.frames.clear()
    frames = []
    start = time.time()
    while time.time() - start < DURATION:
        if processor.frames:
            frames.extend(processor.frames)
            processor.frames.clear()
        time.sleep(0.05)
    filename = "test_audio.wav"
    filename, pcm_resampled = save_audio(frames, filename)
    if filename is None:
        st.error("No audio captured. Check WebRTC connection.")
    else:
        st.success(f"Saved {filename}")
        st.audio(filename, format="audio/wav")

        # Waveform
        fig, ax = plt.subplots()
        ax.plot(pcm_resampled)
        ax.set_title("Captured Audio Waveform")
        st.pyplot(fig)

        # Whisper
        with open(filename, "rb") as f:
            transcript = openai.audio.transcriptions.create(
                model="whisper-1",
                file=f
            )
        st.write("**Whisper Transcription:**")
        st.write(transcript.text)
