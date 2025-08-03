import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import math
from scipy.signal.windows import hamming
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase

# Page Config
st.set_page_config(page_title="Speech & Music Analyzer", page_icon="🎵", layout="wide")
st.title("🎵 Real-Time Speech & Music Analyzer (Web Version)")

fs = 44100  # Sampling rate

# -------------------------------
# Audio Processor
# -------------------------------
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_frames = []

    def recv_audio(self, frame):
        audio_array = frame.to_ndarray().flatten().astype(np.float32)
        self.audio_frames.extend(audio_array)
        return frame

# -------------------------------
# DSP Functions
# -------------------------------
def plot_waveform(audio):
    fig, ax = plt.subplots()
    ax.plot(audio, color="purple")
    ax.set_title("Waveform")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

def plot_spectrogram(audio):
    S = librosa.stft(audio)
    S_db = librosa.amplitude_to_db(abs(S))
    fig, ax = plt.subplots()
    img = librosa.display.specshow(S_db, sr=fs, x_axis='time', y_axis='hz', ax=ax, cmap='magma')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title("Spectrogram")
    st.pyplot(fig)

def detect_pitch(audio):
    pitches, magnitudes = librosa.piptrack(y=audio, sr=fs)
    pitch = pitches[magnitudes.argmax()]
    return pitch

def hz_to_note(freq):
    if freq <= 0:
        return "Unknown"
    A4 = 440
    semitones = 12 * math.log2(freq / A4)
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    note_num = int(round(semitones)) % 12
    return note_names[note_num]

def get_formants(audio):
    audio = np.append(audio, np.zeros(512))
    N = len(audio)
    window = hamming(N)
    audio_win = audio * window
    ncoeff = 2 + fs // 1000
    A = librosa.lpc(audio_win, order=ncoeff)
    rts = np.roots(A)
    rts = [r for r in rts if np.imag(r) >= 0]
    angz = np.arctan2(np.imag(rts), np.real(rts))
    formants = sorted(angz * (fs / (2 * np.pi)))
    return formants[:3]

def classify_audio(audio):
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
    centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=fs))
    return "Music" if zcr > 0.1 and centroid > 2000 else "Speech"

# -------------------------------
# WebRTC Stream
# -------------------------------
ctx = webrtc_streamer(
    key="audio",
    mode=WebRtcMode.SENDRECV,
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"audio": True, "video": False},
)

if ctx.audio_processor:
    if st.button("📊 Analyze Audio"):
        audio_data = np.array(ctx.audio_processor.audio_frames)

        if len(audio_data) > 0:
            st.audio(audio_data, sample_rate=fs)

            col1, col2 = st.columns(2)
            with col1:
                plot_waveform(audio_data)
            with col2:
                plot_spectrogram(audio_data)

            pitch = detect_pitch(audio_data)
            st.write(f"**Pitch:** {pitch:.2f} Hz")
            st.write(f"**Note:** {hz_to_note(pitch)}")
            st.write(f"**Type Detected:** {classify_audio(audio_data)}")

            formants = get_formants(audio_data)
            st.write(f"**Formants (Hz):** {np.round(formants, 2)}")
        else:
            st.warning("No audio captured yet. Please speak or play music before analyzing.")
