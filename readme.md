# Interpretable Audio Understanding: Explainable Environmental Sound Classification

This project implements an **Explainable Environmental Sound Classification (ESC)** system that not only recognizes sounds but also provides transparent insights into *why* the model made its decision. By leveraging state-of-the-art **Explainable AI (XAI)** techniques, we bridge the gap between complex Deep Learning models and human-interpretable results.

## 🌟 Key Features

*   **Deep Learning Classification**: Robust 2D CNN model trained on the **UrbanSound8K** dataset.
*   **Grad-CAM Explanations**: Visualizes spatial attention heatmaps over Mel Spectrograms to show where the model is "looking."
*   **SHAP Feature Importance**: Uses game-theoretic SHAP values to assign importance to specific pixels/regions in the audio features.
*   **Interactive Streamlit Dashboard**: User-friendly interface for uploading audio files and viewing real-time predictions and explanations.
*   **Human-Readable Insights**: Automatically translates technical heatmaps into descriptive text (e.g., "High-frequency transients detected").
*   **Noise Robustness Analysis**: Experiments documenting model behavior under environmental noise conditions.

## 🛠️ Project Structure

```text
├── Explainable_ESC.ipynb    # Research notebook for model training, evaluation, and XAI experiments
├── generate_notebook.py     # Script to generate the comprehensive research notebook
├── project/
│   ├── app.py               # Main Streamlit application entry point
│   ├── utils.py             # Core logic for audio processing, model inference, and XAI generators
│   ├── requirements.txt     # Python dependencies
│   └── model/
│       └── cnn_model.h5     # Pre-trained CNN model weights
└── Dataset/                 # (External) UrbanSound8K dataset directory
```

## 🚀 Getting Started

### 1. Installation

Ensure you have Python 3.8+ installed. Clone this repository and install the required dependencies:

```bash
pip install -r project/requirements.txt
```

### 2. Running the Application

Launch the interactive dashboard using Streamlit:

```bash
streamlit run project/app.py
```

### 3. Exploring the Research

For a deep dive into the model architecture, training process, and detailed XAI experiments (like Grad-CAM and SHAP implementation details), open the Jupyter notebook:

```bash
jupyter notebook Explainable_ESC.ipynb
```

## 🧠 How It Works

1.  **Audio Preprocessing**: Raw audio (.wav) is loaded using `librosa`, trimmed of silence, and normalized.
2.  **Feature Extraction**: The 1D audio waveform is converted into a 2D **Log-Mel Spectrogram**, capturing the time-frequency signature of the sound.
3.  **Inference**: A 2D CNN processes the spectrogram to predict one of 10 categories (e.g., *dog bark*, *siren*, *air conditioner*).
4.  **Explanation Generation**:
    *   **Grad-CAM**: Computes gradients of the target class with respect to the last convolutional layer.
    *   **SHAP**: Explains predictions by comparing them to a background distribution of audio samples.
5.  **Visualization**: The results are presented in a unified dashboard showing the waveform, spectrogram, and explanation overlays.

## 📊 Supported Classes

The model is trained on the UrbanSound8K dataset, supporting the following 10 classes:
- Air Conditioner
- Car Horn
- Children Playing
- Dog Bark
- Drilling
- Engine Idling
- Gun Shot
- Jackhammer
- Siren
- Street Music

---
*Developed for Interpretable Audio Understanding.*
