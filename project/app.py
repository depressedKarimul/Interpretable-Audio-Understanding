import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import cv2
import os
import tempfile
import sys
import tensorflow as tf

# Ensure local imports work regardless of launch directory
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from utils import (  # noqa: E402
    load_audio_file,
    convert_to_mel_spectrogram,
    make_prediction,
    generate_gradcam_heatmap,
    generate_shap_explanation,
    produce_human_readable_explanation,
)

st.set_page_config(page_title="Explainable Environmental Sound Classification", layout="wide")

st.title("Explainable Environmental Sound Classification")
st.caption("Upload an audio clip, get a prediction, and inspect model attention on the spectrogram.")

# Load model
@st.cache_resource
def load_trained_model():
    # Resolve model path relative to this file so it works
    # regardless of where Streamlit is launched from.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "model", "cnn_model.h5")
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    else:
        # Graceful fallback for UI testing without the model
        return None

model = load_trained_model()

if model is None:
    st.error("Model not found. Please place the trained model at 'project/model/cnn_model.h5' relative to the repository root.")
    st.stop()

MODEL_H = int(model.input_shape[1]) if getattr(model, "input_shape", None) else 128
MODEL_W = int(model.input_shape[2]) if getattr(model, "input_shape", None) else 173

with st.sidebar:
    st.header("Controls")
    explain_gradcam = st.toggle("Grad-CAM", value=True)
    explain_shap = st.toggle("SHAP (model-agnostic, slower)", value=False)
    top_db = st.slider("Silence trim aggressiveness (top_db)", 10, 60, 20, 5)
    conf_warn = st.slider("Low-confidence warning threshold", 0.10, 0.95, 0.60, 0.05)
    st.divider()
    st.caption(f"Model input: {MODEL_H}×{MODEL_W} (Mel×Frames)")

st.header("Upload audio")
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg", "flac"])

if uploaded_file is not None and model is not None:
    st.audio(uploaded_file.getvalue())

    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp:
        tmp.write(uploaded_file.getvalue())
        temp_path = tmp.name

    try:
        with st.spinner("Processing audio..."):
            y_norm, sr = load_audio_file(temp_path, target_sr=22050)
            if y_norm is None or len(y_norm) == 0:
                st.warning("The uploaded file appears to be mostly silence after trimming. Try a louder / clearer clip.")
                st.stop()

            mel_spec = convert_to_mel_spectrogram(y_norm, sr, n_mels=MODEL_H, max_pad_len=MODEL_W)
            pred_label, confidence, input_data = make_prediction(model, mel_spec)

        tabs = st.tabs(["Result", "Visualizations", "Explanations", "About"])

        with tabs[0]:
            st.subheader("Prediction")
            c1, c2, c3 = st.columns([1, 1, 2])
            with c1:
                st.metric("Predicted class", pred_label.replace("_", " ").title())
            with c2:
                st.metric("Confidence", f"{confidence*100:.1f}%")
            with c3:
                if confidence < conf_warn:
                    st.warning("Low confidence. The clip may be ambiguous or out-of-distribution.")

            probs = model.predict(input_data.astype(np.float32), verbose=0)[0]
            topk = 5
            top_idx = np.argsort(-probs)[:topk]
            top_labels = [pred_label] * 0  # placeholder to keep lint quiet (not used)
            try:
                from utils import CLASSES
                top_labels = [CLASSES[i].replace("_", " ").title() for i in top_idx]
            except Exception:
                top_labels = [str(i) for i in top_idx]
            top_vals = probs[top_idx]

            figp, axp = plt.subplots(figsize=(7, 3))
            axp.barh(top_labels[::-1], top_vals[::-1])
            axp.set_xlabel("Probability")
            axp.set_xlim(0, 1)
            axp.grid(True, axis="x", alpha=0.25)
            st.pyplot(figp)

        with tabs[1]:
            st.subheader("Waveform and spectrogram")
            col1, col2 = st.columns(2)
            with col1:
                fig1, ax1 = plt.subplots(figsize=(7, 3))
                librosa.display.waveshow(y_norm, sr=sr, ax=ax1)
                ax1.set_title("Waveform")
                ax1.set_xlabel("Time (s)")
                st.pyplot(fig1)

            with col2:
                fig2, ax2 = plt.subplots(figsize=(7, 3))
                img = librosa.display.specshow(
                    mel_spec, sr=sr, hop_length=512, x_axis="time", y_axis="mel", fmax=8000, ax=ax2
                )
                fig2.colorbar(img, ax=ax2, format="%+2.0f dB")
                ax2.set_title("Log-Mel spectrogram")
                st.pyplot(fig2)

        heatmap_resized = None
        shap_spec = None
        time_freq_region = None
        text_explanation = None

        with tabs[2]:
            st.subheader("Explanations")
            if not explain_gradcam and not explain_shap:
                st.info("Enable Grad-CAM and/or SHAP in the sidebar to generate explanations.")
            else:
                if explain_gradcam:
                    with st.spinner("Computing Grad-CAM..."):
                        heatmap = generate_gradcam_heatmap(model, input_data)
                        heatmap_resized = cv2.resize(heatmap, (input_data.shape[2], input_data.shape[1]))
                        time_freq_region, text_explanation = produce_human_readable_explanation(pred_label, heatmap_resized)

                    colg1, colg2 = st.columns(2)
                    with colg1:
                        fig3, ax3 = plt.subplots(figsize=(7, 3))
                        ax3.imshow(mel_spec, aspect="auto", cmap="magma", origin="lower")
                        ax3.imshow(heatmap_resized, aspect="auto", cmap="jet", alpha=0.5, origin="lower")
                        ax3.set_title("Grad-CAM overlay")
                        ax3.axis("off")
                        st.pyplot(fig3)
                    with colg2:
                        st.markdown("**Human-readable summary**")
                        st.write(f"**Important region:** {time_freq_region}")
                        st.write(f"**Explanation:** {text_explanation}")

                if explain_shap:
                    with st.spinner("Computing SHAP (this can be slow)..."):
                        shap_spec = generate_shap_explanation(model, input_data)
                    fig4, ax4 = plt.subplots(figsize=(7, 3))
                    vmax = float(np.max(np.abs(shap_spec))) if shap_spec is not None else 0.0
                    if vmax <= 0:
                        vmax = 1.0
                    im4 = ax4.imshow(shap_spec, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax, origin="lower")
                    fig4.colorbar(im4, ax=ax4, label="SHAP value")
                    ax4.set_title("SHAP feature importance (model-agnostic)")
                    ax4.axis("off")
                    st.pyplot(fig4)

        with tabs[3]:
            st.markdown(
                """
This app runs a CNN trained on UrbanSound8K-style Mel spectrograms and provides optional explainability:

- **Grad-CAM**: highlights time-frequency regions most responsible for the predicted class.
- **SHAP**: model-agnostic masked SHAP (slower, but stable with modern TensorFlow).
"""
            )

    finally:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass
