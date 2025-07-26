import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av

st.title("WebRTC Connection Debug")

class AudioProcessor:
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        print("Audio frame received")
        return frame

ctx = webrtc_streamer(
    key="debug",
    mode=WebRtcMode.SENDRECV,
    audio_receiver_size=256,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True,
    audio_processor_factory=AudioProcessor,
)

if ctx.state.playing:
    st.write("**Connected & streaming audio!**")
else:
    st.write("Waiting for connection...")
