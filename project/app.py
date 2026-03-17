import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import cv2
import os
import tempfile
import sys
import tensorflow as tf
import pandas as pd

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
    generate_graph_explanation,
    generate_final_conclusion,
)

st.set_page_config(page_title="Explainable Environmental Sound Classification", layout="wide")

APP_TITLE = "Explainable Environmental Sound Classification"
st.markdown(f"## {APP_TITLE}")
st.write(
    "This research-demo app classifies environmental sounds using a deep learning model and provides "
    "explainability visualizations (e.g., Grad-CAM) to show which time–frequency regions drive predictions."
)

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
        return None

model = load_trained_model()

def _repo_root():
    return os.path.abspath(os.path.join(_HERE, ".."))


@st.cache_data
def _load_urbansound_metadata():
    csv_path = os.path.join(_repo_root(), "Dataset", "UrbanSound8K.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None


df_meta = _load_urbansound_metadata()

MODEL_H = int(model.input_shape[1]) if model is not None and getattr(model, "input_shape", None) else 128
MODEL_W = int(model.input_shape[2]) if model is not None and getattr(model, "input_shape", None) else 173

CLASSES = None
try:
    from utils import CLASSES as _CLASSES  # type: ignore
    CLASSES = list(_CLASSES)
except Exception:
    CLASSES = None


def _example_audio_path_for_class(class_name: str) -> str | None:
    """Try to find an example file from the UrbanSound8K folder structure, if present locally."""
    if df_meta is None:
        return None
    # UrbanSound8K audio files live under Dataset/fold{fold}/slice_file_name
    subset = df_meta[df_meta["class"] == class_name]
    if subset.empty:
        return None
    for _, row in subset.head(50).iterrows():
        candidate = os.path.join(_repo_root(), "Dataset", f"fold{int(row['fold'])}", str(row["slice_file_name"]))
        if os.path.exists(candidate):
            return candidate
    return None


with st.sidebar:
    st.header("Sidebar")

    st.subheader("Audio input")
    uploaded_file = st.file_uploader("Upload an audio file (.wav)", type=["wav"])

    st.caption("Examples (requires UrbanSound8K audio files under `Dataset/fold*/...`)")  # buttons required by spec
    ex_col1, ex_col2 = st.columns(2)
    with ex_col1:
        ex_dog = st.button("Dog bark", width="stretch")
        ex_drill = st.button("Drilling", width="stretch")
    with ex_col2:
        ex_siren = st.button("Siren", width="stretch")
        ex_gun = st.button("Gun shot", width="stretch")

    st.divider()
    st.subheader("Model settings")
    sr = st.number_input("Sample rate", min_value=8000, max_value=48000, value=22050, step=1000)
    duration_s = st.number_input("Audio duration (seconds)", min_value=1.0, max_value=10.0, value=4.0, step=0.5)

    st.caption("Mel spectrogram parameters")
    n_mels = st.number_input("n_mels", min_value=32, max_value=256, value=int(MODEL_H), step=16)
    hop_length = st.number_input("hop_length", min_value=128, max_value=2048, value=512, step=128)
    n_fft = st.number_input("n_fft", min_value=256, max_value=8192, value=2048, step=256)
    fmax = st.number_input("fmax (Hz)", min_value=2000, max_value=22050, value=8000, step=500)

    st.divider()
    st.subheader("Explainability")
    explain_gradcam = st.toggle("Grad-CAM", value=True)
    explain_shap = st.toggle("SHAP (optional, slower)", value=False)
    trim_top_db = st.slider("Trim silence (top_db)", 10, 60, 20, 5)
    conf_warn = st.slider("Low-confidence warning threshold", 0.10, 0.95, 0.60, 0.05)

    st.divider()
    if model is None:
        st.warning("Model not found at `project/model/cnn_model.h5`. Demo will run without predictions.")
    else:
        st.caption(f"Loaded model input: {MODEL_H}×{MODEL_W} (Mel×Frames)")

def _load_audio_from_source() -> tuple[str | None, bytes | None]:
    """Return (path, bytes) for either uploaded file or an example selection."""
    # Example buttons override uploader when clicked.
    example_map = {
        "dog_bark": ex_dog,
        "siren": ex_siren,
        "drilling": ex_drill,
        "gun_shot": ex_gun,
    }
    for k, clicked in example_map.items():
        if clicked:
            p = _example_audio_path_for_class(k)
            if p is None:
                st.sidebar.error("Example audio not found locally. Add UrbanSound8K audio under `Dataset/fold*/...`.")
                return None, None
            with open(p, "rb") as f:
                return p, f.read()

    if uploaded_file is None:
        return None, None
    return uploaded_file.name, uploaded_file.getvalue()


src_name, src_bytes = _load_audio_from_source()

page = st.tabs(["Demo Dashboard", "Model Performance", "Dataset Info"])

with page[0]:
    st.markdown("### Audio preview")
    if src_bytes is None:
        st.info("Upload a `.wav` file in the sidebar or click an example button to start.")
    else:
        st.audio(src_bytes)

        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{os.path.basename(src_name or 'audio.wav')}") as tmp:
            tmp.write(src_bytes)
            temp_path = tmp.name

        try:
            with st.spinner("Loading + preprocessing audio..."):
                y_norm, sr_eff = load_audio_file(
                    temp_path,
                    target_sr=int(sr),
                    duration_s=float(duration_s),
                    top_db=int(trim_top_db),
                )
                if y_norm is None or len(y_norm) == 0:
                    st.warning("This clip becomes mostly silence after trimming. Try a louder / clearer sample.")
                    st.stop()

            # Compute mel_spec and predictions earlier so the AI can explain the early graphs
            mel_spec = convert_to_mel_spectrogram(
                y_norm,
                sr_eff,
                n_mels=int(n_mels),
                hop_length=int(hop_length),
                n_fft=int(n_fft),
                fmax=float(fmax),
                max_pad_len=int(MODEL_W),
            )
            input_data = mel_spec.reshape(1, mel_spec.shape[0], mel_spec.shape[1], 1).astype(np.float32)

            pred_label = "Unknown"
            probs = None
            confidence = 0.0
            if model is not None:
                pred_label, confidence, _ = make_prediction(model, mel_spec)
                probs = model.predict(input_data, verbose=0)[0]

            # Waveform
            wf_container = st.container(border=True)
            with wf_container:
                st.markdown("### Waveform")
                fig_w, ax_w = plt.subplots(figsize=(10, 3))
                librosa.display.waveshow(y_norm, sr=sr_eff, ax=ax_w)
                ax_w.set_xlabel("Time (s)")
                ax_w.grid(True, alpha=0.25)
                st.pyplot(fig_w)
                
                if pred_label != "Unknown":
                    with st.spinner("AI is analyzing the waveform..."):
                        exp_w = generate_graph_explanation("Waveform (Amplitude vs Time)", pred_label, "This graph displays the raw audio amplitude envelope over time.")
                        if exp_w: st.info(f"**AI Analysis:** {exp_w}")

            # Spectrogram
            spec_container = st.container(border=True)
            with spec_container:
                st.markdown("### Mel spectrogram (model input)")
                fig_s, ax_s = plt.subplots(figsize=(10, 3))
                im = librosa.display.specshow(
                    mel_spec,
                    sr=sr_eff,
                    hop_length=int(hop_length),
                    x_axis="time",
                    y_axis="mel",
                    fmax=float(fmax),
                    ax=ax_s,
                )
                fig_s.colorbar(im, ax=ax_s, format="%+2.0f dB")
                ax_s.set_title("Log-Mel spectrogram")
                st.pyplot(fig_s)
                
                if pred_label != "Unknown":
                    with st.spinner("AI is analyzing the spectrogram..."):
                        exp_s = generate_graph_explanation("Mel Spectrogram (Frequency vs Time)", pred_label, "This graph displays the frequency energy patterns (low to high bands) of the sound.")
                        if exp_s: st.info(f"**AI Analysis:** {exp_s}")

            # Prediction section
            pred_container = st.container(border=True)
            with pred_container:
                st.markdown("### Prediction")
                if model is None:
                    st.info("Model is not available, so predictions are disabled. Add `project/model/cnn_model.h5`.")
                else:
                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric("Predicted class", pred_label.replace("_", " ").title())
                    with c2:
                        st.metric("Confidence", f"{confidence*100:.1f}%")
                        if confidence < float(conf_warn):
                            st.warning("Low confidence. This clip may be ambiguous or out-of-distribution.")

                    labels = CLASSES if CLASSES is not None else [f"class_{i}" for i in range(len(probs))]
                    fig_p, ax_p = plt.subplots(figsize=(10, 3))
                    ax_p.bar(range(len(probs)), probs)
                    ax_p.set_xticks(range(len(probs)))
                    ax_p.set_xticklabels([l.replace("_", " ") for l in labels], rotation=35, ha="right")
                    ax_p.set_ylim(0, 1)
                    ax_p.set_ylabel("Probability")
                    ax_p.set_title("Class probabilities")
                    ax_p.grid(True, axis="y", alpha=0.25)
                    st.pyplot(fig_p)
                    
                    with st.spinner("AI is analyzing the class probabilities..."):
                        top_indices = np.argsort(probs)[-3:][::-1]
                        top_classes = [labels[i].replace('_', ' ') for i in top_indices]
                        top_probs = [probs[i] for i in top_indices]
                        probs_str = ", ".join([f"{cls} ({p*100:.1f}%)" for cls, p in zip(top_classes, top_probs)])
                        exp_p = generate_graph_explanation("Class Probabilities Bar Chart", pred_label, f"The top predictions are: {probs_str}. Explain why the model is confident or confused between these.")
                        if exp_p: st.info(f"**AI Analysis:** {exp_p}")

            # Explainability section
            exp_container = st.container(border=True)
            with exp_container:
                st.markdown("### Explainability (Grad-CAM)")
                if model is None:
                    st.info("Model not available. Grad-CAM is disabled.")
                elif not explain_gradcam:
                    st.info("Enable Grad-CAM in the sidebar to generate the heatmap overlay.")
                else:
                    with st.spinner("Computing Grad-CAM..."):
                        heatmap = generate_gradcam_heatmap(model, input_data)
                        heatmap_resized = cv2.resize(heatmap, (input_data.shape[2], input_data.shape[1]))

                    col_e1, col_e2 = st.columns([2, 1])
                    with col_e1:
                        fig_g, ax_g = plt.subplots(figsize=(10, 3))
                        ax_g.imshow(mel_spec, aspect="auto", cmap="magma", origin="lower")
                        ax_g.imshow(heatmap_resized, aspect="auto", cmap="jet", alpha=0.5, origin="lower")
                        ax_g.set_title("Grad-CAM overlay on Mel spectrogram")
                        ax_g.axis("off")
                        st.pyplot(fig_g)
                    with col_e2:
                        if model is not None:
                            time_freq_region, text_explanation = produce_human_readable_explanation(
                                pred_label if "pred_label" in locals() else "unknown",
                                heatmap_resized,
                            )
                            st.markdown("**Interpretation**")
                            st.write(f"**Important region:** {time_freq_region}")
                            st.write(f"**Explanation:** {text_explanation}")

                if explain_shap and model is not None:
                    st.markdown("### SHAP (optional)")
                    with st.spinner("Computing SHAP (may take time)..."):
                        shap_spec = generate_shap_explanation(model, input_data)
                    fig_sh, ax_sh = plt.subplots(figsize=(10, 3))
                    vmax = float(np.max(np.abs(shap_spec))) if shap_spec is not None else 0.0
                    if vmax <= 0:
                        vmax = 1.0
                    im_sh = ax_sh.imshow(shap_spec, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax, origin="lower")
                    fig_sh.colorbar(im_sh, ax=ax_sh, label="SHAP value")
                    ax_sh.set_title("SHAP feature importance (masked, model-agnostic)")
                    ax_sh.axis("off")
                    st.pyplot(fig_sh)
                    
                    if pred_label != "Unknown" and pred_label is not None:
                        with st.spinner("AI is analyzing the SHAP visualization..."):
                            exp_sh = generate_graph_explanation("SHAP Feature Importance Map", pred_label, "This graph highlights which subtle parts of the audio contributed most significantly to the final decision.")
                            if exp_sh: st.info(f"**AI Analysis:** {exp_sh}")

            # Final Conclusion section
            if pred_label != "Unknown" and model is not None:
                final_container = st.container(border=True)
                with final_container:
                    st.markdown("### Final AI Conclusion")
                    with st.spinner("AI is summarizing the final decision..."):
                        top_indices = np.argsort(probs)[-3:][::-1]
                        top_classes = [labels[i].replace('_', ' ') for i in top_indices]
                        top_probs = [probs[i] for i in top_indices]
                        probs_str = ", ".join([f"{cls} ({p*100:.1f}%)" for cls, p in zip(top_classes, top_probs)])
                        
                        final_conclusion = generate_final_conclusion(pred_label, confidence, probs_str)
                        if final_conclusion:
                            st.success(f"**Verdict:** {final_conclusion}")
                        else:
                            st.write("Could not generate a final AI conclusion at this time.")

        finally:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception:
                pass

with page[1]:
    st.markdown("### Model performance")
    st.write(
        "This section summarizes training/evaluation results (accuracy, confusion matrix, and training curves). "
        "If you generated figures from the notebook, place them in the repo root as:"
        " `paper_fig_explanations.png`, `paper_fig_insertion_deletion.png`, `paper_fig_model_comparison.png`."
    )
    fig_paths = {
        "Model comparison": os.path.join(_repo_root(), "paper_fig_model_comparison.png"),
        "Insertion/Deletion curves": os.path.join(_repo_root(), "paper_fig_insertion_deletion.png"),
        "Explainability overview": os.path.join(_repo_root(), "paper_fig_explanations.png"),
    }
    shown_any = False
    cols = st.columns(3)
    for (title, path), col in zip(fig_paths.items(), cols):
        with col:
            st.markdown(f"**{title}**")
            if os.path.exists(path):
                st.image(path, width="stretch")
                shown_any = True
            else:
                st.caption("Figure not found.")
    if not shown_any:
        st.info("No saved performance figures found yet. Run the notebook section that saves paper-ready figures.")

    # If metadata exists, show class distribution as a lightweight dataset-derived “performance context”
    if df_meta is not None:
        st.markdown("#### Dataset class distribution (from metadata)")
        vc = df_meta["class"].value_counts()
        figd, axd = plt.subplots(figsize=(10, 3))
        axd.bar(range(len(vc.index)), vc.values)
        axd.set_xticks(range(len(vc.index)))
        axd.set_xticklabels(vc.index, rotation=35, ha="right")
        axd.set_ylabel("Count")
        axd.grid(True, axis="y", alpha=0.25)
        st.pyplot(figd)

with page[2]:
    st.markdown("### UrbanSound8K dataset information")
    if df_meta is None:
        st.warning("Could not find `Dataset/UrbanSound8K.csv`. Add it to enable dataset info.")
    else:
        n_classes = int(df_meta["classID"].nunique())
        n_samples = int(len(df_meta))
        st.metric("Number of classes", n_classes)
        st.metric("Total audio samples", n_samples)
        st.markdown("#### Example class labels")
        st.write(sorted(df_meta["class"].unique().tolist()))
