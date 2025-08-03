import streamlit as st
import sounddevice as sd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import math
from scipy.signal import hamming

# -------------------------------
# 🎯 App Title
# -------------------------------
st.set_page_config(page_title="Speech & Music Analyzer", page_icon="🎵", layout="wide")
st.title("🎵 Real-Time Speech & Music Analyzer")
st.markdown("Analyze **speech** and **music** live: waveform, spectrogram, pitch, note, formants & classification.")

# -------------------------------
# 🎤 Record Audio
# -------------------------------
fs = 44100  # Sampling rate

def record_audio(duration=3):
    st.info("Recording... Speak or play music now!")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    st.success("Recording complete ✅")
    return np.squeeze(audio)

# -------------------------------
# 📊 Waveform
# -------------------------------
def plot_waveform(audio):
    fig, ax = plt.subplots()
    ax.plot(audio, color="purple")
    ax.set_title("Waveform")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

# -------------------------------
# 📊 Spectrogram
# -------------------------------
def plot_spectrogram(audio):
    S = librosa.stft(audio)
    S_db = librosa.amplitude_to_db(abs(S))
    fig, ax = plt.subplots()
    img = librosa.display.specshow(S_db, sr=fs, x_axis='time', y_axis='hz', ax=ax, cmap='magma')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title("Spectrogram")
    st.pyplot(fig)

# -------------------------------
# 🎼 Pitch Detection
# -------------------------------
def detect_pitch(audio):
    pitches, magnitudes = librosa.piptrack(y=audio, sr=fs)
    pitch = pitches[magnitudes.argmax()]
    return pitch

# -------------------------------
# 🎹 Note Mapping
# -------------------------------
def hz_to_note(freq):
    if freq <= 0: 
        return "Unknown"
    A4 = 440
    semitones = 12 * math.log2(freq / A4)
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    note_num = int(round(semitones)) % 12
    return note_names[note_num]

# -------------------------------
# 🗣 Formant Detection (Speech)
# -------------------------------
def get_formants(audio):
    audio = np.append(audio, np.zeros(512))  # padding
    N = len(audio)
    window = hamming(N)
    audio_win = audio * window

    ncoeff = 2 + fs // 1000
    A = librosa.lpc(audio_win, order=ncoeff)
    rts = np.roots(A)
    rts = [r for r in rts if np.imag(r) >= 0]

    angz = np.arctan2(np.imag(rts), np.real(rts))
    formants = sorted(angz * (fs / (2 * np.pi)))
    return formants[:3]  # First three formants

# -------------------------------
# 🎵 Speech/Music Classification
# -------------------------------
def classify_audio(audio):
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
    centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=fs))
    if zcr > 0.1 and centroid > 2000:
        return "Music"
    else:
        return "Speech"

# -------------------------------
# 🚀 Main App Logic
# -------------------------------
duration = st.slider("Recording Duration (seconds)", 1, 10, 3)

if st.button("🎤 Record & Analyze"):
    audio = record_audio(duration)
    st.audio(audio, sample_rate=fs)

    col1, col2 = st.columns(2)

    with col1:
        plot_waveform(audio)

    with col2:
        plot_spectrogram(audio)

    pitch = detect_pitch(audio)
    st.write(f"**Pitch:** {pitch:.2f} Hz")
    st.write(f"**Note:** {hz_to_note(pitch)}")
    st.write(f"**Type Detected:** {classify_audio(audio)}")

    # Formants
    formants = get_formants(audio)
    st.write(f"**Formants (Hz):** {np.round(formants, 2)}")
