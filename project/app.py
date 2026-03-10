import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import cv2
import os
import tensorflow as tf
from utils import (
    load_audio_file,
    convert_to_mel_spectrogram,
    make_prediction,
    generate_gradcam_heatmap,
    generate_shap_explanation,
    produce_human_readable_explanation
)

st.set_page_config(page_title="Explainable Environmental Sound Classification", layout="wide")

st.title("Explainable Environmental Sound Classification")

# Load model
@st.cache_resource
def load_trained_model():
    model_path = os.path.join("model", "cnn_model.h5")
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    else:
        # Graceful fallback for UI testing without the model
        return None

model = load_trained_model()

if model is None:
    st.error(f"Model not found at 'model/cnn_model.h5'. Please place your trained model there.")

st.header("Upload Audio")
uploaded_file = st.file_uploader("Upload an audio file (.wav)", type=["wav"])

if uploaded_file is not None and model is not None:
    # Save the file temporarily
    temp_path = "temp_audio.wav"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    st.audio(temp_path, format='audio/wav')
    
    with st.spinner("Processing audio..."):
        # Process audio
        y_norm, sr = load_audio_file(temp_path)
        mel_spec = convert_to_mel_spectrogram(y_norm, sr)
        
        # Prediction
        pred_label, confidence, input_data = make_prediction(model, mel_spec)
        
    with st.spinner("Generating Explanations (Grad-CAM & SHAP)..."):
        # Explanations
        heatmap = generate_gradcam_heatmap(model, input_data)
        heatmap_resized = cv2.resize(heatmap, (input_data.shape[2], input_data.shape[1]))
        
        shap_spec = generate_shap_explanation(model, input_data)
        time_freq_region, text_explanation = produce_human_readable_explanation(pred_label, heatmap_resized)
        
        # Section: Prediction Result
        st.header("Prediction Result")
        st.write(f"**Prediction:** {pred_label.replace('_', ' ').capitalize()}")
        st.write(f"**Confidence:** {confidence * 100:.1f}%")
        
        # Section: Model Explanation
        st.header("Model Explanation")
        st.write("**Important region:**")
        st.write(f"{time_freq_region}")
        st.write("")
        st.write("**Explanation:**")
        st.write(f"{text_explanation}")
        
        # Section: Visualization Panel
        st.header("Visualization Panel")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Waveform Visualization")
            fig1, ax1 = plt.subplots(figsize=(6, 3))
            librosa.display.waveshow(y_norm, sr=sr, ax=ax1)
            ax1.set(title="Audio Waveform")
            st.pyplot(fig1)
            
            st.subheader("Mel Spectrogram")
            fig2, ax2 = plt.subplots(figsize=(6, 3))
            img = librosa.display.specshow(mel_spec, sr=sr, hop_length=512, x_axis='time', y_axis='mel', fmax=8000, ax=ax2)
            fig2.colorbar(img, ax=ax2, format='%+2.0f dB')
            ax2.set(title="Log-Mel Spectrogram")
            st.pyplot(fig2)
            
        with col2:
            st.subheader("Grad-CAM Heatmap")
            fig3, ax3 = plt.subplots(figsize=(6, 3))
            # using aspect='auto' and proper origin
            ax3.imshow(mel_spec, aspect='auto', cmap='magma', origin='lower')
            ax3.imshow(heatmap_resized, aspect='auto', cmap='jet', alpha=0.5, origin='lower')
            ax3.set(title="Grad-CAM Overlay")
            ax3.axis('off')
            st.pyplot(fig3)
            
            st.subheader("SHAP Feature Importance")
            fig4, ax4 = plt.subplots(figsize=(6, 3))
            vmax = np.max(np.abs(shap_spec))
            if vmax == 0:  # in case shap fails and returns zeros
                vmax = 1
            im4 = ax4.imshow(shap_spec, aspect='auto', cmap='coolwarm', 
                             vmin=-vmax, vmax=vmax, origin='lower')
            fig4.colorbar(im4, ax=ax4, label='SHAP Value')
            ax4.set(title="SHAP Values")
            ax4.axis('off')
            st.pyplot(fig4)
            
    # Cleanup temporarily saved file
    if os.path.exists(temp_path):
        os.remove(temp_path)
