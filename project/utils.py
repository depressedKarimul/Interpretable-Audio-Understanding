import os
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import shap
import matplotlib.pyplot as plt

# Classes for UrbanSound8K
CLASSES = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling',
           'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']

def load_audio_file(file_path, target_sr=22050):
    """Loads, resamples, trims silence, and normalizes audio.

    Returns (y_normalized, sr). If the effective audio after trimming is extremely
    short (near-silence), y_normalized will be an empty array.
    """
    y, sr = librosa.load(file_path, sr=target_sr, mono=True)
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)

    # If almost no audio remains after trimming, treat as silence / invalid input.
    if y_trimmed.size == 0:
        return np.array([]), sr

    max_val = np.max(np.abs(y_trimmed))
    if max_val > 0:
        y_normalized = y_trimmed / max_val
    else:
        y_normalized = y_trimmed
    return y_normalized, sr

def convert_to_mel_spectrogram(y, sr, n_mels=128, hop_length=512, fmax=8000, max_pad_len=174):
    """Converts audio array into a padded Log-Mel Spectrogram."""
    if y is None or y.size == 0:
        # Return an all-zero spectrogram if input is effectively silence
        return np.zeros((n_mels, max_pad_len))
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length, fmax=fmax)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    if S_dB.shape[1] > max_pad_len:
        S_dB = S_dB[:, :max_pad_len]
    else:
        pad_width = max_pad_len - S_dB.shape[1]
        S_dB = np.pad(S_dB, pad_width=((0, 0), (0, pad_width)), mode='constant')
        
    return S_dB

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
        class_idx = tf.argmax(preds[0])
        class_channel = preds[:, class_idx]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()

def generate_shap_explanation(model, img_array, background_size=10):
    try:
        # Generate a small background of random noise instead of all zeros for better SHAP gradients
        # Use values close to actual mel spectrogram ranges (-80 to 0)
        background = np.random.uniform(low=-80, high=0, size=(background_size, 128, 174, 1))
        
        # We use GradientExplainer or DeepExplainer. Give preference to GradientExplainer for safety in TF2+
        # But instructions said DeepExplainer. For a TF2 model, DeepExplainer might complain if run eagerly.
        explainer = shap.DeepExplainer(model, background)
        shap_values = explainer.shap_values(img_array)
        
        predicted_class_idx = np.argmax(model.predict(img_array, verbose=0))
        
        if isinstance(shap_values, list):
            shap_spec = shap_values[predicted_class_idx][0, :, :, 0]
        else:
            if len(shap_values.shape) == 5: # (1, 128, 174, 1, 10)
                shap_spec = shap_values[0, ..., predicted_class_idx][:, :, 0]
            else:
                shap_spec = shap_values[0, ..., predicted_class_idx]
                if len(shap_spec.shape) == 3:
                    shap_spec = shap_spec[:, :, 0]
                    
        return shap_spec
    except Exception as e:
        print(f"SHAP explanation failed: {e}")
        # Return fallback dummy shap values
        return np.zeros((128, 174))

def produce_human_readable_explanation(pred_label, heatmap):
    # Calculate properties of the heatmap
    time_mean = np.mean(heatmap, axis=0)
    mel_mean = np.mean(heatmap, axis=1)
    
    highest_time_zone = np.argmax(time_mean) 
    highest_freq_zone = np.argmax(mel_mean)
    
    time_sec = (highest_time_zone / 174) * 4.0 # approximate max length
    
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
    
    explanation_text = reasons.get(pred_label, 'Detected characteristic frequency profiles associated with this class.')
    
    return explanation_time_freq, explanation_text
