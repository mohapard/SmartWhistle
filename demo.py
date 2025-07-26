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
device_id = st.text_input("Enter deviceId (see above). Leave empty for default:")
media_constraints = {"audio": True, "video": True}
if device_id.strip():
    media_constraints = {"audio": {"deviceId": {"exact": device_id}}, "video": True}

# WebRTC
webrtc_ctx = webrtc_streamer(
    key="mic-debug",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    audio_receiver_size=256,
    media_stream_constraints=media_constraints,
    async_processing=True,
    audio_processor_factory=AudioProcessor,
)

# Status
if webrtc_ctx.state.playing:
    st.success("Connected & streaming.")
else:
    st.warning("Waiting for connection...")

# Audio frames
if webrtc_ctx.audio_processor:
    st.write(f"**Audio frames received:** {webrtc_ctx.audio_processor.count}")
    fig, ax = plt.subplots()
    ax.plot(webrtc_ctx.audio_processor.last_audio)
    ax.set_ylim([-32768, 32768])
    ax.set_title("Live Audio Waveform")
    st.pyplot(fig)
