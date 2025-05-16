import streamlit as st
import numpy as np
import librosa
import os
import tensorflow as tf
from pydub import AudioSegment
import tempfile

# Load the trained model
model = tf.keras.models.load_model("audio_deepfake_detector1.keras")

# --- Handcrafted Feature Extraction ---
def analyze_audio_traits(file_path):
    y, sr = librosa.load(file_path, sr=16000)

    pitches, _ = librosa.piptrack(y=y, sr=sr)
    pitches = pitches[pitches > 0]
    pitch_std = np.std(pitches) if len(pitches) > 0 else 0

    spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y)) if len(y) > 0 else 0

    intervals = librosa.effects.split(y, top_db=20)
    total_speech_duration = np.sum([(end - start) for start, end in intervals]) / sr
    pause_ratio = 1 - (total_speech_duration / (len(y) / sr))

    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
    rms = np.mean(librosa.feature.rms(y=y))

    return {
        "pitch_std": pitch_std,
        "spectral_flatness": spectral_flatness,
        "pause_ratio": pause_ratio,
        "zcr": zcr,
        "rms": rms
    }

# --- Explanation Generator ---
def generate_unique_explanation(features, model_prediction):
    result = "🔍 The model predicts this audio is **FAKE**." if model_prediction == 1 else "✅ The model predicts this audio is **REAL**."
    explanation = []

    # 1. Pitch Variation
    if features["pitch_std"] > 30:
        if model_prediction == 1:
            explanation.append(f"- High pitch variability (±{features['pitch_std']:.2f} Hz) indicates unnatural fluctuations — a known artifact in AI-generated voices.")
        else:
            explanation.append(f"- Pitch variability (±{features['pitch_std']:.2f} Hz) reflects natural emotional expression in human speech.")
    else:
        if model_prediction == 1:
            explanation.append(f"- Low pitch variability (±{features['pitch_std']:.2f} Hz) shows a flat tone, typical of synthetic voices lacking emotion.")
        else:
            explanation.append(f"- Pitch variability remains within normal human range (±{features['pitch_std']:.2f} Hz).")

    # 2. Spectral Flatness
    if features["spectral_flatness"] > 0.3:
        if model_prediction == 1:
            explanation.append(f"- High spectral flatness ({features['spectral_flatness']:.2f}) indicates robotic tone, supporting the synthetic nature of the voice.")
        else:
            explanation.append(f"- Despite higher spectral flatness ({features['spectral_flatness']:.2f}), speech characteristics align with natural speaking.")
    else:
        explanation.append(f"- Spectral richness detected ({features['spectral_flatness']:.2f}), resembling typical human vocal patterns.")

    # 3. Pause Ratio
    if features["pause_ratio"] > 0.2:
        if model_prediction == 1:
            explanation.append(f"- Unnatural pause distribution (Pause ratio: {features['pause_ratio']*100:.1f}%) often results from auto-generated speech alignment.")
        else:
            explanation.append(f"- Though the pause ratio is {features['pause_ratio']*100:.1f}%, the flow resembles real expressive speech.")
    else:
        if model_prediction == 1:
            explanation.append(f"- Smooth flow (Pause ratio: {features['pause_ratio']*100:.1f}%) might be artificially optimized, not matching natural breathing pauses.")
        else:
            explanation.append(f"- Low pause ratio ({features['pause_ratio']*100:.1f}%) indicates fluent, human-like delivery.")

    # 4. Zero-Crossing Rate (ZCR)
    if features["zcr"] > 0.1:
        if model_prediction == 1:
            explanation.append(f"- High zero-crossing rate ({features['zcr']:.2f}) implies excessive signal activity — possibly synthetic noise artifacts.")
        else:
            explanation.append(f"- ZCR ({features['zcr']:.2f}) is consistent with fast or energetic speech seen in natural human conversations.")
    else:
        if model_prediction == 1:
            explanation.append(f"- Low ZCR ({features['zcr']:.2f}) may reflect overly smooth or flat audio — a subtle synthetic indicator.")
        else:
            explanation.append(f"- Low ZCR ({features['zcr']:.2f}) matches expected human speech patterns.")

    # 5. RMS Energy
    if features["rms"] < 0.02:
        if model_prediction == 1:
            explanation.append(f"- Very low energy (RMS: {features['rms']:.3f}) suggests lifeless delivery — a common trait of deepfake audio.")
        else:
            explanation.append(f"- Low energy (RMS: {features['rms']:.3f}) might indicate soft-spoken or intentionally quiet human speech.")
    else:
        if model_prediction == 1:
            explanation.append(f"- Energy levels (RMS: {features['rms']:.3f}) appear normal, but other acoustic patterns suggest synthetic origin.")
        else:
            explanation.append(f"- Adequate energy (RMS: {features['rms']:.3f}) suggests clarity and expressiveness, common in real speech.")

    return result + "\n\n" + "\n".join(explanation)



# --- Prediction using Model ---
def predict_with_model(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    max_pad_len = 174

    if mfcc.shape[1] < max_pad_len:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_pad_len - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]

    mfcc = mfcc.reshape(1, 40, 174, 1).astype(np.float32)
    prediction_score = model.predict(mfcc)[0][0]
    return 1 if prediction_score >= 0.4 else 0, prediction_score

# --- Streamlit App ---
st.title("🎧 AI Deepfake Audio Detector")
st.write("Upload an audio file to detect whether it's **real or AI-generated**, along with explainable insights.")

uploaded_file = st.file_uploader("Choose an audio file (MP3 or WAV)", type=["mp3", "wav"])

if uploaded_file is not None:
    try:
        # Convert audio using pydub if it's MP3
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
            if uploaded_file.name.endswith(".mp3"):
                audio = AudioSegment.from_file(uploaded_file, format="mp3")
                audio.export(temp_wav.name, format="wav")
            else:
                temp_wav.write(uploaded_file.read())

            st.audio(temp_wav.name)
            st.write("🔎 Analyzing audio...")

            features = analyze_audio_traits(temp_wav.name)
            prediction, confidence = predict_with_model(temp_wav.name)
            explanation = generate_unique_explanation(features, prediction)

            st.markdown(explanation)
            st.markdown(f"🧠 **Model confidence:** `{confidence:.4f}`")

    except Exception as e:
        st.error(f"Something went wrong: {str(e)}")
    finally:
        if os.path.exists(temp_wav.name):
            os.remove(temp_wav.name)
