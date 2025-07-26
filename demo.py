import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av, time, numpy as np, soundfile as sf, librosa, matplotlib.pyplot as plt

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
DURATION = 3
TARGET_RATE = 16000

def save_audio(frames, filename, input_rate=48000, target_rate=TARGET_RATE):
    audio_bytes = b''.join(frames)
    if not audio_bytes:
        print("[DEBUG] No audio captured.")
        return None, None
    # FIX: Use proper PCM16 data directly
    pcm = np.frombuffer(audio_bytes, dtype=np.int16)
    # If stereo → mono
    if pcm.ndim > 1:
        pcm = np.mean(pcm, axis=1)
    # Normalize to float32 for resampling
    pcm = pcm.astype(np.float32) / 32768.0
    # Resample to target rate
    pcm_resampled = librosa.resample(pcm, orig_sr=input_rate, target_sr=target_rate)
    # Back to int16 for saving
    pcm_int16 = np.int16(np.clip(pcm_resampled, -1.0, 1.0) * 32767)
    sf.write(filename, pcm_int16, target_rate, subtype="PCM_16")
    return filename, pcm_resampled

class AudioProcessor:
    def __init__(self):
        self.frames = []
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        # FIX: Force signed 16‑bit interleaved PCM
        arr = frame.to_ndarray(format="s16")
        self.frames.append(arr.tobytes())
        return frame

st.title("Static Audio Fix Test")
st.write("Record 3s → Save → Playback → Waveform")

webrtc_ctx = webrtc_streamer(
    key="fix_test",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True,
    audio_processor_factory=AudioProcessor,
    sendback_audio=False
)

if webrtc_ctx.audio_processor and st.button("Record & Play"):
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

    filename = "test_fixed.wav"
    filename, pcm_resampled = save_audio(frames, filename)
    if filename:
        st.success("Saved and processed")
        st.audio(filename, format="audio/wav")

        # Waveform
        fig, ax = plt.subplots()
        ax.plot(pcm_resampled)
        ax.set_title("Waveform")
        st.pyplot(fig)
    else:
        st.error("No audio captured.")

