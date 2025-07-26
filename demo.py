import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

class DebugAudioProcessor:
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        arr = frame.to_ndarray()
        print(f"[DEBUG] Frame: shape={arr.shape}, dtype={arr.dtype}, pts={frame.pts}")
        return frame

st.title("WebRTC Audio Debugger")
st.write("This will log every incoming audio frame in the console.")

webrtc_ctx = webrtc_streamer(
    key="debugger",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"audio": True},
    async_processing=True,
    audio_processor_factory=DebugAudioProcessor,
    sendback_audio=False
)
