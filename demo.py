import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

class DebugAudioProcessor:
    def __init__(self):
        self.frames_received = 0
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        self.frames_received += 1
        print(f"[DEBUG] Got frame #{self.frames_received} shape={frame.to_ndarray().shape}")
        return frame

st.title("WebRTC Audio Debugger")

status = st.empty()

webrtc_ctx = webrtc_streamer(
    key="debugger",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True,
    audio_processor_factory=DebugAudioProcessor,
    sendback_audio=False
)

if webrtc_ctx.audio_processor:
    st.write("Mic connected. Start speaking or whistling...")
    st.write(f"Frames received: {webrtc_ctx.audio_processor.frames_received}")
else:
    st.write("Waiting for mic connection...")
