import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av, numpy as np
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

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

# --- AUDIO PROCESSOR ---
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
st.title("Mic Debug with Auto-Refresh")

# Auto-refresh every 1s
st.experimental_autorefresh(interval=1000, key="refresh")

# Show available devices
st.markdown("#### Available Audio Devices in Your Browser")
components.html("""
<script>
navigator.mediaDevices.enumerateDevices().then(devices => {
  const audioDevices = devices.filter(d => d.kind === 'audioinput');
  const pre = document.createElement('pre');
  pre.textContent = JSON.stringify(audioDevices, null, 2);
  document.body.appendChild(pre);
});
</script>
""", height=300)

# Mic selection
device_id = st.text_input("Enter deviceId (see above). Leave emp
