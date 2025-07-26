import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av, numpy as np
import matplotlib.pyplot as plt

# --- TURN/STUN CONFIG ---
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

# --- AUDIO PROCESSOR (DEBUG) ---
class AudioProcessor:
    def __init__(self):
        self.frames = []
        self.last_audio = np.zeros(1024)
        self.count = 0
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        self.count += 1
        pcm = frame.to_ndarray().astype(np.int16).flatten()
        self.last_audio = pcm
        self.frames.append(pcm.tobytes())
        return frame

# --- UI ---
st.title("Microphone Debug with Waveform & Device Selector")

# Device selection
st.markdown("#### Select Microphone Device")
device_id = st.text_input("Optional: Enter exact device ID (leave empty for default mic)")

# Build media constraints
media_constraints = {"audio": True, "video": True}  # dummy video track
if device_id.strip():
    media_constraints = {
        "audio": {"deviceId": {"exact": device_id}},
        "video": True
    }

# Start WebRTC
webrtc_ctx = webrtc_streamer(
    key="mic-debug",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    audio_receiver_size=256,
    media_stream_constraints=media_constraints,
    async_processing=True,
    audio_processor_factory=AudioProcessor,
)

# Connection state
if webrtc_ctx.state.playing:
    st.success("Connected & streaming.")
else:
    st.warning("Waiting for connection... (Check permissions or try Chrome)")

# Audio debug info
if webrtc_ctx.audio_processor:
    st.write(f"**Audio frames received:** {webrtc_ctx.audio_processor.count}")

    # Waveform visualization
    fig, ax = plt.subplots()
    ax.plot(webrtc_ctx.audio_processor.last_audio)
    ax.set_ylim([-32768, 32768])
    ax.set_title("Live Audio Waveform")
    st.pyplot(fig)

st.markdown("""
**Instructions:**  
1. Make sure Chrome is used (best for WebRTC).  
2. Select the correct microphone in the widget **or enter its device ID** above (you can find device IDs using `navigator.mediaDevices.enumerateDevices()` in browser console).  
3. Speak or blow a whistle — if audio works, the **frame counter will increase** and the **waveform will move**.  
4. If frame count stays at 0:  
   - Check macOS **System Preferences → Security & Privacy → Microphone** → Ensure your browser/Python has mic access.  
   - Try entering a specific device ID.  
   - Test on another browser or network.
""")
