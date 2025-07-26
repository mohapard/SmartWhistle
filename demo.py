import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

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

# --- SESSION STATE ---
if "frame_count" not in st.session_state:
    st.session_state.frame_count = 0
if "status" not in st.session_state:
    st.session_state.status = "Idle"

# --- AUDIO PROCESSOR (DEBUG) ---
class AudioProcessor:
    def __init__(self):
        self.frames = []
        self.count = 0
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        self.count += 1
        self.frames.append(frame.to_ndarray().tobytes())
        return frame

# --- UI ---
st.title("Microphone Debug App (TURN-enabled)")
st.write("This will show if audio frames are actually coming in.")

webrtc_ctx = webrtc_streamer(
    key="mic-debug",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    audio_receiver_size=256,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True,
    audio_processor_factory=AudioProcessor,
)

# --- DEBUG OUTPUT ---
if webrtc_ctx.state.playing:
    st.success("Connected & streaming (browser reports active connection).")
else:
    st.warning("Waiting for connection... (check mic permissions & HTTPS)")

# Show live frame count
if webrtc_ctx.audio_processor:
    st.session_state.frame_count = webrtc_ctx.audio_processor.count
st.write(f"**Audio frames received:** {st.session_state.frame_count}")

# Add note for user
st.markdown("""
**Instructions:**  
1. Select your microphone in the dropdown.  
2. Speak or blow a whistle — if the frame counter increases, audio is flowing.  
3. If it stays at 0:  
   - Ensure browser mic permissions are allowed (padlock icon → allow mic).  
   - Try Chrome (best support).  
   - Select the correct input device.  
""")
