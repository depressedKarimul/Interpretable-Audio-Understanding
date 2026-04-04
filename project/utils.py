import os
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import shap
import matplotlib.pyplot as plt

try:
    from groq import Groq
except ImportError:
    Groq = None

def get_groq_api_key():
    key = os.environ.get("GROQ_API_KEY")
    if key:
        return key
    
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('GROQ_API_KEY='):
                    return line.split('=', 1)[1].strip()
                elif line.startswith("gsk_"):
                    return line
    return None

api_key = get_groq_api_key()
groq_client = Groq(api_key=api_key) if Groq and api_key else None

# Classes for UrbanSound8K
CLASSES = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling',
           'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']

SR = 22050
DURATION = 4.0
HOP_LENGTH = 512
TARGET_LEN = int(SR * DURATION)
SPEC_FRAMES = 1 + (TARGET_LEN // HOP_LENGTH)  # 173 with these defaults

def load_audio_file(file_path, target_sr=SR, duration_s=DURATION, top_db=20):
    """Loads, resamples, trims silence, and normalizes audio.

    Returns (y_normalized, sr). If the effective audio after trimming is extremely
    short (near-silence), y_normalized will be an empty array.
    """
    y, sr = librosa.load(file_path, sr=target_sr, mono=True)
    y_trimmed, _ = librosa.effects.trim(y, top_db=int(top_db))

    # If almost no audio remains after trimming, treat as silence / invalid input.
    if y_trimmed.size == 0:
        return np.array([]), sr

    # Enforce a fixed duration (pad/trim) for stable downstream features
    target_len = int(float(duration_s) * sr)
    if y_trimmed.size < target_len:
        y_trimmed = np.pad(y_trimmed, (0, target_len - y_trimmed.size), mode="constant")
    elif y_trimmed.size > target_len:
        y_trimmed = y_trimmed[:target_len]

    max_val = np.max(np.abs(y_trimmed))
    if max_val > 0:
        y_normalized = y_trimmed / max_val
    else:
        y_normalized = y_trimmed
    return y_normalized, sr

def convert_to_mel_spectrogram(
    y,
    sr,
    n_mels=128,
    hop_length=HOP_LENGTH,
    n_fft=2048,
    fmax=8000,
    max_pad_len=None,
):
    """Converts audio array into a fixed-width Log-Mel Spectrogram."""
    if max_pad_len is None:
        max_pad_len = SPEC_FRAMES
    if y is None or y.size == 0:
        # Return an all-zero spectrogram if input is effectively silence
        return np.zeros((n_mels, max_pad_len))

    # Safety pad to avoid n_fft warnings on extremely short inputs
    if y.size < int(n_fft):
        y = np.pad(y, (0, int(n_fft) - y.size), mode="constant")

    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=int(n_mels),
        hop_length=int(hop_length),
        n_fft=int(n_fft),
        fmax=float(fmax),
        power=2.0,
    )
    S_dB = librosa.power_to_db(S, ref=np.max)

    # Fix time axis to a consistent width.
    S_dB = librosa.util.fix_length(S_dB, size=int(max_pad_len), axis=1)
    return S_dB.astype(np.float32)

def make_prediction(model, mel_spec):
    """Run a forward pass and return (label, confidence, input_data).

    If the model is highly uncertain (low softmax probability), callers can
    use the confidence value to flag an unreliable prediction.
    """
    # Reshape for model input
    input_data = mel_spec.reshape(1, mel_spec.shape[0], mel_spec.shape[1], 1)
    preds = model.predict(input_data, verbose=0)
    class_idx = np.argmax(preds[0])
    confidence = float(preds[0][class_idx])

    return CLASSES[class_idx], confidence, input_data

def get_last_conv_layer_name(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    # fallback to the last layer if no conv2d found, though error might be better
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4:
            return layer.name
    raise ValueError("No matching Conv2D layer found in the model.")

def generate_gradcam_heatmap(model, img_array, last_conv_layer_name=None):
    if last_conv_layer_name is None:
        last_conv_layer_name = get_last_conv_layer_name(model)
        
    _ = model.predict(img_array, verbose=0)
    grad_model = tf.keras.models.Model(
        inputs=model.inputs, 
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        # Handle case where model.output is a list
        if isinstance(preds, list):
            preds = preds[0]
        class_idx = tf.argmax(preds[0])
        class_channel = preds[:, class_idx]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()

def generate_shap_explanation(model, img_array, max_evals=200):
    """Model-agnostic SHAP that won't break TF gradients."""
    try:
        img_array = np.array(img_array, dtype=np.float32)
        if img_array.ndim != 4:
            raise ValueError(f"Expected img_array with shape (1,H,W,1); got shape {img_array.shape}")

        H, W = int(img_array.shape[1]), int(img_array.shape[2])
        pred = model.predict(img_array, verbose=0)[0]
        predicted_class_idx = int(np.argmax(pred))

        def _predict_fn(x):
            x = np.array(x, dtype=np.float32)
            if x.ndim == 3:
                x = x[..., np.newaxis]
            return model.predict(x, verbose=0)

        masker = shap.maskers.Image("blur(8,8)", (H, W, 1))
        explainer = shap.Explainer(_predict_fn, masker)
        sv = explainer(img_array, max_evals=int(max_evals), batch_size=20)

        vals = getattr(sv, "values", None)
        if vals is None:
            return np.zeros((H, W), dtype=np.float32)

        vals = np.array(vals)
        if vals.ndim == 5 and vals.shape[-1] > predicted_class_idx:
            return vals[0, :, :, 0, predicted_class_idx].astype(np.float32)
        if vals.ndim == 4:
            return vals[0, :, :, 0].astype(np.float32)
        return np.zeros((H, W), dtype=np.float32)
    except Exception as e:
        print(f"SHAP explanation failed: {e}")
        h = int(img_array.shape[1]) if hasattr(img_array, "shape") and len(img_array.shape) >= 3 else 128
        w = int(img_array.shape[2]) if hasattr(img_array, "shape") and len(img_array.shape) >= 3 else SPEC_FRAMES
        return np.zeros((h, w), dtype=np.float32)

def produce_human_readable_explanation(pred_label, heatmap):
    # Calculate properties of the heatmap
    time_mean = np.mean(heatmap, axis=0)
    mel_mean = np.mean(heatmap, axis=1)
    
    highest_time_zone = np.argmax(time_mean) 
    highest_freq_zone = np.argmax(mel_mean)
    
    width = int(heatmap.shape[1]) if hasattr(heatmap, "shape") else 1
    time_sec = (highest_time_zone / max(1, width)) * DURATION # approximate max length
    
    if highest_freq_zone > 80:
        freq_desc = "high-frequency band"
    elif highest_freq_zone > 40:
        freq_desc = "mid-frequency band"

def get_groq_api_key():
    key = os.environ.get("GROQ_API_KEY")
    if key:
        return key
    
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('GROQ_API_KEY='):
                    return line.split('=', 1)[1].strip()
                elif line.startswith("gsk_"):
                    return line
    return None

api_key = get_groq_api_key()
groq_client = Groq(api_key=api_key) if Groq and api_key else None

# Classes for UrbanSound8K
CLASSES = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling',
           'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']

SR = 22050
DURATION = 4.0
HOP_LENGTH = 512
TARGET_LEN = int(SR * DURATION)
SPEC_FRAMES = 1 + (TARGET_LEN // HOP_LENGTH)  # 173 with these defaults

def load_audio_file(file_path, target_sr=SR, duration_s=DURATION, top_db=20):
    """Loads, resamples, trims silence, and normalizes audio.

    Returns (y_normalized, sr). If the effective audio after trimming is extremely
    short (near-silence), y_normalized will be an empty array.
    """
    y, sr = librosa.load(file_path, sr=target_sr, mono=True)
    y_trimmed, _ = librosa.effects.trim(y, top_db=int(top_db))

    # If almost no audio remains after trimming, treat as silence / invalid input.
    if y_trimmed.size == 0:
        return np.array([]), sr

    # Enforce a fixed duration (pad/trim) for stable downstream features
    target_len = int(float(duration_s) * sr)
    if y_trimmed.size < target_len:
        y_trimmed = np.pad(y_trimmed, (0, target_len - y_trimmed.size), mode="constant")
    elif y_trimmed.size > target_len:
        y_trimmed = y_trimmed[:target_len]

    max_val = np.max(np.abs(y_trimmed))
    if max_val > 0:
        y_normalized = y_trimmed / max_val
    else:
        y_normalized = y_trimmed
    return y_normalized, sr

def convert_to_mel_spectrogram(
    y,
    sr,
    n_mels=128,
    hop_length=HOP_LENGTH,
    n_fft=2048,
    fmax=8000,
    max_pad_len=None,
):
    """Converts audio array into a fixed-width Log-Mel Spectrogram."""
    if max_pad_len is None:
        max_pad_len = SPEC_FRAMES
    if y is None or y.size == 0:
        # Return an all-zero spectrogram if input is effectively silence
        return np.zeros((n_mels, max_pad_len))

    # Safety pad to avoid n_fft warnings on extremely short inputs
    if y.size < int(n_fft):
        y = np.pad(y, (0, int(n_fft) - y.size), mode="constant")

    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=int(n_mels),
        hop_length=int(hop_length),
        n_fft=int(n_fft),
        fmax=float(fmax),
        power=2.0,
    )
    S_dB = librosa.power_to_db(S, ref=np.max)

    # Fix time axis to a consistent width.
    S_dB = librosa.util.fix_length(S_dB, size=int(max_pad_len), axis=1)
    return S_dB.astype(np.float32)

def make_prediction(model, mel_spec):
    """Run a forward pass and return (label, confidence, input_data).

    If the model is highly uncertain (low softmax probability), callers can
    use the confidence value to flag an unreliable prediction.
    """
    # Reshape for model input
    input_data = mel_spec.reshape(1, mel_spec.shape[0], mel_spec.shape[1], 1)
    preds = model.predict(input_data, verbose=0)
    class_idx = np.argmax(preds[0])
    confidence = float(preds[0][class_idx])

    return CLASSES[class_idx], confidence, input_data

def get_last_conv_layer_name(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    # fallback to the last layer if no conv2d found, though error might be better
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4:
            return layer.name
    raise ValueError("No matching Conv2D layer found in the model.")

def generate_gradcam_heatmap(model, img_array, last_conv_layer_name=None):
    if last_conv_layer_name is None:
        last_conv_layer_name = get_last_conv_layer_name(model)
        
    _ = model.predict(img_array, verbose=0)
    grad_model = tf.keras.models.Model(
        inputs=model.inputs, 
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        # Handle case where model.output is a list
        if isinstance(preds, list):
            preds = preds[0]
        class_idx = tf.argmax(preds[0])
        class_channel = preds[:, class_idx]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()

def generate_shap_explanation(model, img_array, max_evals=200):
    """Model-agnostic SHAP that won't break TF gradients."""
    try:
        img_array = np.array(img_array, dtype=np.float32)
        if img_array.ndim != 4:
            raise ValueError(f"Expected img_array with shape (1,H,W,1); got shape {img_array.shape}")

        H, W = int(img_array.shape[1]), int(img_array.shape[2])
        pred = model.predict(img_array, verbose=0)[0]
        predicted_class_idx = int(np.argmax(pred))

        def _predict_fn(x):
            x = np.array(x, dtype=np.float32)
            if x.ndim == 3:
                x = x[..., np.newaxis]
            return model.predict(x, verbose=0)

        masker = shap.maskers.Image("blur(8,8)", (H, W, 1))
        explainer = shap.Explainer(_predict_fn, masker)
        sv = explainer(img_array, max_evals=int(max_evals), batch_size=20)

        vals = getattr(sv, "values", None)
        if vals is None:
            return np.zeros((H, W), dtype=np.float32)

        vals = np.array(vals)
        if vals.ndim == 5 and vals.shape[-1] > predicted_class_idx:
            return vals[0, :, :, 0, predicted_class_idx].astype(np.float32)
        if vals.ndim == 4:
            return vals[0, :, :, 0].astype(np.float32)
        return np.zeros((H, W), dtype=np.float32)
    except Exception as e:
        print(f"SHAP explanation failed: {e}")
        h = int(img_array.shape[1]) if hasattr(img_array, "shape") and len(img_array.shape) >= 3 else 128
        w = int(img_array.shape[2]) if hasattr(img_array, "shape") and len(img_array.shape) >= 3 else SPEC_FRAMES
        return np.zeros((h, w), dtype=np.float32)

def generate_integrated_gradients(model, img_array, baseline=None, steps=50):
    """Computes Integrated Gradients for a given spatial input."""
    try:
        if baseline is None:
            baseline = np.zeros_like(img_array).astype(np.float32)
        img_array = img_array.astype(np.float32)
        baseline = baseline.astype(np.float32)
        target_idx = np.argmax(model.predict(img_array, verbose=0)[0])
        alphas = np.linspace(0, 1, steps + 1)
        grads = []
        for a in alphas:
            interp = baseline + a * (img_array - baseline)
            interp = tf.convert_to_tensor(interp)
            with tf.GradientTape() as tape:
                tape.watch(interp)
                preds = model(interp)
                target = preds[0, target_idx]
            g = tape.gradient(target, interp)
            grads.append(g.numpy())
        grads = np.stack(grads, axis=0)
        avg_grads = (grads[:-1] + grads[1:]) / 2
        integrated = (img_array - baseline) * np.mean(avg_grads, axis=0)
        
        ig_attr = np.squeeze(integrated)
        if ig_attr.ndim == 3:
            ig_attr = ig_attr[:, :, 0]
        
        ig_attr = np.maximum(ig_attr, 0)
        return ig_attr
    except Exception as e:
        print(f"Integrated Gradients failed: {e}")
        h = int(img_array.shape[1]) if hasattr(img_array, "shape") and len(img_array.shape) >= 3 else 128
        w = int(img_array.shape[2]) if hasattr(img_array, "shape") and len(img_array.shape) >= 3 else SPEC_FRAMES
        return np.zeros((h, w), dtype=np.float32)

from scipy.ndimage import gaussian_filter

def calculate_ecs(gradcam, shap_map, ig_map, k_percent=30):
    """Calculates Explainability Consistency Score given normalized heatmaps."""
    try:
        def top_k_mask(saliency, k):
            sal_flat = np.abs(saliency).flatten()
            if len(sal_flat) == 0: return np.zeros_like(saliency)
            thresh = np.percentile(sal_flat, 100 - k)
            return (np.abs(saliency) >= thresh).astype(np.float32)

        def iou(m1, m2):
            m1, m2 = m1.flatten(), m2.flatten()
            inter = np.sum(m1 * m2)
            union = np.sum(m1) + np.sum(m2) - inter
            return float(inter / union) if union > 0 else 0.0

        # Apply Gaussian smoothing and calculate top-k masks
        gradcam_smooth = gaussian_filter(gradcam, sigma=2)
        shap_smooth = gaussian_filter(shap_map, sigma=2)
        ig_smooth = gaussian_filter(ig_map, sigma=2)

        gradcam_bin = top_k_mask(gradcam_smooth, k_percent)
        shap_bin = top_k_mask(shap_smooth, k_percent)
        ig_bin = top_k_mask(ig_smooth, k_percent)

        iou_gc_shap = iou(gradcam_bin, shap_bin)
        iou_gc_ig = iou(gradcam_bin, ig_bin)
        iou_shap_ig = iou(shap_bin, ig_bin)
        ecs = (iou_gc_shap + iou_gc_ig + iou_shap_ig) / 3
        
        return {
            "ecs": ecs,
            "iou_gc_shap": iou_gc_shap,
            "iou_gc_ig": iou_gc_ig,
            "iou_shap_ig": iou_shap_ig
        }
    except Exception as e:
        print(f"ECS calculation failed: {e}")
        return None

def produce_human_readable_explanation(pred_label, heatmap):
    # Calculate properties of the heatmap
    time_mean = np.mean(heatmap, axis=0)
    mel_mean = np.mean(heatmap, axis=1)
    
    highest_time_zone = np.argmax(time_mean) 
    highest_freq_zone = np.argmax(mel_mean)
    
    width = int(heatmap.shape[1]) if hasattr(heatmap, "shape") else 1
    time_sec = (highest_time_zone / max(1, width)) * DURATION # approximate max length
    
    if highest_freq_zone > 80:
        freq_desc = "high-frequency band"
    elif highest_freq_zone > 40:
        freq_desc = "mid-frequency band"
    else:
        freq_desc = "low-frequency band"
        
    explanation_time_freq = f"{time_sec:.1f} - {time_sec + 0.5:.1f} seconds ({freq_desc})"
    
    reasons = {
        'siren': 'Periodic high-pitched waveform typical of sirens.',
        'dog_bark': 'Brief broadband transients representing animal vocalization.',
        'drilling': 'Continuous, intense low/mid-frequency noise from mechanical vibration.',
        'children_playing': 'Intermittent high-pitched harmonics resembling voices/shouts.',
        'engine_idling': 'Very low-frequency, sustained periodic rumble.',
        'air_conditioner': 'Continuous low-frequency hum characteristic of mechanical appliances.',
        'car_horn': 'Loud, sustained tonal bursts in mid to high frequencies.',
        'gun_shot': 'Extremely brief, intense, broad-frequency impulse.',
        'jackhammer': 'Rapid, periodic wide-band impulses from pneumatic impacts.',
        'street_music': 'Complex mix of harmonic and percussive features typical of instruments.'
    }
    
    base_explanation = reasons.get(pred_label, 'Detected characteristic frequency profiles associated with this class.')
    explanation_text = base_explanation

    # Enhance with Groq if available
    if groq_client:
        prompt = (
            f"You are an AI assistant helping a user understand a deep learning model's prediction for an audio clip. "
            f"The model has classified the environmental sound as '{pred_label}'. "
            f"The Grad-CAM heatmap visualization indicates that the most important region for this prediction "
            f"is between {time_sec:.1f} - {time_sec + 0.5:.1f} seconds in the {freq_desc}. "
            f"A technical baseline reason is: '{base_explanation}'\n\n"
            f"Please write a VERY brief, human-like (1-2 sentences) and highly intuitive explanation interpreting this graph to a non-technical user. "
            f"Don't just repeat the technical metrics, frame it like you are simply pointing out what part of the audio they should listen to and why it sounds like a {pred_label.replace('_', ' ')}."
        )
        try:
            chat_completion = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",  # Used updated available model on Groq
                temperature=0.7,
            )
            explanation_text = chat_completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"Groq API error: {e}")
            pass # fallback to technical base explanation
            
    return explanation_time_freq, explanation_text

def generate_graph_explanation(graph_type, pred_label, extra_context=""):
    """Uses Groq to explain what a specific graph represents for a given prediction."""
    if not groq_client: return None
    
    prompt = (
        f"You are an AI assistant helping a user understand a deep learning model's prediction for an audio clip. "
        f"The model has classified the environmental sound as '{pred_label}'.\n\n"
        f"You are tasked with explaining the '{graph_type}' visualization to a non-technical user. "
        f"{extra_context}\n"
        f"Provide a brief (1-2 sentences), highly intuitive, and human-like explanation. "
        f"Explain what they are generally seeing in this type of graph and why it might look the way it does for a {pred_label.replace('_', ' ')}. Do not use overly complex jargon."
    )
    
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            temperature=0.7,
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Groq API error for general explanation: {e}")
        return None

def generate_final_conclusion(pred_label, confidence, probs_str):
    """Uses Groq to summarize the final model decision."""
    if not groq_client: return None
    
    prompt = (
        f"You are an AI assistant tasked with summarizing a deep learning model's final prediction for an environmental audio clip.\n\n"
        f"The model's final chosen label is '{pred_label}' with a confidence of {confidence*100:.1f}%.\n"
        f"The top probabilities were: {probs_str}.\n\n"
        f"In 2-3 sentences, briefly sum up why the model arrived at this conclusion based on everything you've explained about the audio waveforms and frequency distributions earlier. "
        f"Speak naturally and directly to the user (e.g., 'Ultimately, the model is highly confident this is a dog bark because...'). Avoid highly technical jargon."
    )

    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            temperature=0.7,
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Groq API error for final conclusion: {e}")
        return None

def generate_lime_explanation(model, img_array, num_samples=50):
    try:
        from lime import lime_image
        from skimage.segmentation import slic
        
        def predict_fn_lime(images):
            # LIME provides RGB images of shape (N, H, W, 3).
            # Convert back to (N, H, W, 1) expected by the model.
            x = np.array(images)[:, :, :, 0:1]
            return model.predict(x, batch_size=64, verbose=0)
            
        sample_for_lime = np.repeat(img_array[0], 3, axis=-1)
        explainer_lime = lime_image.LimeImageExplainer()
        explanation_lime = explainer_lime.explain_instance(
            sample_for_lime.astype(np.double),
            predict_fn_lime,
            top_labels=1,
            hide_color=0,
            num_samples=num_samples,
            segmentation_fn=lambda x: slic(x[:, :, 0], n_segments=30, compactness=10, channel_axis=None),
        )
        lime_img, lime_mask = explanation_lime.get_image_and_mask(
            explanation_lime.top_labels[0], positive_only=True, num_features=10, hide_rest=False
        )
        return lime_mask.astype(np.float32)
    except Exception as e:
        print(f"LIME explanation failed (ensure lime/skimage are installed): {e}")
        return None

def calculate_faithfulness_score(model, img_array, saliency_map, n_perm=30, top_frac=0.2):
    """Calculates Faithfulness score: the correlation between saliency importance and prediction drop."""
    try:
        h, w = saliency_map.shape
        flat_sal = saliency_map.flatten()
        pred_class = np.argmax(model.predict(img_array, verbose=0)[0])
        base_prob = float(model.predict(img_array, verbose=0)[0, pred_class])
        
        drops, sals = [], []
        for _ in range(n_perm):
            idx = np.random.choice(len(flat_sal), size=int(top_frac * len(flat_sal)), replace=False)
            mask = np.ones(len(flat_sal))
            mask[idx] = 0
            x_masked = (img_array[0].flatten() * mask).reshape(1, h, w, 1).astype(np.float32)
            p_masked = float(model.predict(x_masked, verbose=0)[0, pred_class])
            
            drops.append(base_prob - p_masked)
            sals.append(float(np.mean(flat_sal[idx])))
            
        if np.std(sals) > 0:
            correlation = np.corrcoef(drops, sals)[0, 1]
        else:
            correlation = 0.0
            
        return correlation
    except Exception as e:
        print(f"Faithfulness calculation failed: {e}")
        return None
