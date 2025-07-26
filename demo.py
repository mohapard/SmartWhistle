import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av, time, numpy as np, soundfile as sf, librosa, openai

openai.api_key = st.secrets["openai_key"]

RATE = 44100
DURATION = 3
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

def save_audio(frames, filename, target_rate=16000):
    audio = b''.join(frames)
    pcm = np.frombuffer(audio, dtype=np.int16)
    # If stereo, mix down to mono
    if pcm.ndim > 1:
        pcm = np.mean(pcm, axis=1)
    # Resample to 16kHz
    pcm = librosa.resample(pcm.astype(np.float32), orig_sr=RATE, target_sr=target_rate)
    # Save as 16‑bit PCM
    sf.write(filename, pcm, target_rate, subtype='PCM_16')
    return filename

class AudioProcessor:
    def __init__(self):
        self.frames = []
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        self.frames.append(frame.to_ndarray().astype(np.int16).tobytes())
        return frame

st.title("Audio Capture Test")
st.write("Record 3 seconds and play it back")

webrtc_ctx = webrtc_streamer(
    key="recorder",
    mode=WebRtcMode.SENDONLY,
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
    save_audio(frames, filename)
    st.success(f"Saved {filename}")
    
    # Playback
    st.audio(filename, format="audio/wav")
    
    # Send to Whisper
    with open(filename, "rb") as f:
        transcript = openai.audio.transcriptions.create(
            model="whisper-1",
            file=f
        )
    st.write("**Whisper Transcription:**")
    st.write(transcript.text)
