import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av, time, numpy as np, soundfile as sf, librosa, openai, matplotlib.pyplot as plt

openai.api_key = st.secrets["openai_key"]

INPUT_RATE = 44100
TARGET_RATE = 16000
DURATION = 3
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# --- Correct Audio Saving ---
def save_audio(frames, filename, target_rate=TARGET_RATE, input_rate=INPUT_RATE):
    # Concatenate all frame bytes
    audio_bytes = b''.join(frames)
    # Convert to float32 (correct format from WebRTC)
    pcm = np.frombuffer(audio_bytes, dtype=np.float32)
    # If stereo, mix to mono
    if pcm.ndim > 1:
        pcm = np.mean(pcm, axis=1)
    # Resample to target rate
    pcm_resampled = librosa.resample(pcm, orig_sr=input_rate, target_sr=target_rate)
    # Normalize and convert to int16
    pcm_int16 = np.int16(np.clip(pcm_resampled, -1.0, 1.0) * 32767)
    # Save as WAV
    sf.write(filename, pcm_int16, target_rate, subtype='PCM_16')
    return filename, pcm_resampled

# --- Audio Processor ---
class AudioProcessor:
    def __init__(self):
        self.frames = []
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        # Extract float32 planar audio
        arr = frame.to_ndarray(format="flt")
        self.frames.append(arr.tobytes())
        return frame

# --- UI ---
st.title("Audio Capture Test (Fixed Pipeline)")
st.write("Record 3 seconds, play back the audio, view waveform, and transcribe with Whisper.")

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

    # Save & process
    filename = "test_audio.wav"
    filename, pcm_resampled = save_audio(frames, filename)
    st.success(f"Saved {filename}")

    # Playback
    st.audio(filename, format="audio/wav")

    # Waveform display
    fig, ax = plt.subplots()
    ax.plot(pcm_resampled)
    ax.set_title("Captured Audio Waveform")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

    # Whisper Transcription
    with open(filename, "rb") as f:
        transcript = openai.audio.transcriptions.create(
            model="whisper-1",
            file=f
        )
    st.write("**Whisper Transcription:**")
    st.write(transcript.text)
