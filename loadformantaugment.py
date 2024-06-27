import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import librosa
import parselmouth
from parselmouth.praat import call
from sklearn.preprocessing import StandardScaler
import joblib
from concurrent.futures import ProcessPoolExecutor

def extract_formants_and_hnr(audio_path, n_formants=7):
    snd = parselmouth.Sound(audio_path)
    formant = call(snd, "To Formant (burg)", 0.0, 7.0, 5500, 0.025, 50.0)
    hnr = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)

    formant_features = []
    for i in range(1, n_formants + 1):
        f_values = []
        for t in range(1, formant.get_number_of_frames() + 1):
            f = call(formant, "Get value at time", i, formant.get_time_from_frame_number(t), "Hertz", "Linear")
            if not np.isnan(f):
                f_values.append(f)
        formant_features.append(np.mean(f_values) if f_values else 0)

    # Menghitung nilai rata-rata HNR selama durasi suara
    hnr_values = []
    for t in range(1, hnr.get_number_of_frames() + 1):
        h = call(hnr, "Get value at time", hnr.get_time_from_frame_number(t), "Linear")
        if not np.isnan(h):
            hnr_values.append(h)
    hnr_average = np.mean(hnr_values) if hnr_values else 0

    # Menambahkan HNR ke fitur
    formant_features.append(hnr_average)

    return np.array(formant_features)

def normalize_data(X, scaler=None):
    X_reshaped = X.reshape(X.shape[0], -1)
    if scaler is None:
        scaler = StandardScaler().fit(X_reshaped)
    X_scaled = scaler.transform(X_reshaped)
    return X_scaled.reshape(X.shape[0], X.shape[1], 1), scaler

def load_data(healthy_folder, depressed_folder):
    healthy_paths = [os.path.join(healthy_folder, f) for f in os.listdir(healthy_folder)]
    depressed_paths = [os.path.join(depressed_folder, f) for f in os.listdir(depressed_folder)]
    audio_paths = healthy_paths + depressed_paths
    labels = [0] * len(healthy_paths) + [1] * len(depressed_paths)
    
    with ProcessPoolExecutor() as executor:
        features = list(executor.map(extract_formants_and_hnr, audio_paths))
    
    features = np.array(features)
    features = features.reshape(features.shape[0], features.shape[1], 1)
    return features, np.array(labels), healthy_paths, depressed_paths

def predict_new_audio(audio_path, model_path, scaler_path):
    # Load model
    model = load_model(model_path)
    
    # Load scaler
    scaler = joblib.load(scaler_path)
    
    # Ekstraksi fitur dari file audio baru
    features = extract_formants_and_hnr(audio_path)
    
    # Normalisasi data
    features = features.reshape(1, -1)
    features, _ = normalize_data(features, scaler)
    
    # Prediksi
    prediction = model.predict(features)
    
    # Konversi prediksi ke label
    predicted_label = np.argmax(prediction, axis=1)
    
    return predicted_label, prediction

if __name__ == '__main__':
    # Load data dan cek distribusi
    healthy_folder = "Z:/speech depressed/sample/non depressed"
    depressed_folder = "Z:/speech depressed/depresidata"
    X, Y, healthy_paths, depressed_paths = load_data(healthy_folder, depressed_folder)
    
    # Cek distribusi data
    unique, counts = np.unique(Y, return_counts=True)
    print(f"Data distribution: {dict(zip(unique, counts))}")
    
    # Lakukan prediksi pada beberapa sampel dari dataset
    model_path = "Z:/speech depressed/tesfix.h5"
    scaler_path = 'tesfix.pkl'
    sample_paths = [healthy_paths[3], healthy_paths[4], healthy_paths[5],healthy_paths[6],healthy_paths[0],healthy_paths[1],healthy_paths[2], depressed_paths[125], depressed_paths[201], depressed_paths[79]]  # Path ke file audio yang akan diprediksi
    for sample_path in sample_paths:
        predicted_label, prediction = predict_new_audio(sample_path, model_path, scaler_path)
        label_map = {0: "Sehat", 1: "depressed"}
        result = label_map[predicted_label[0]]
        print(f"Predicted Label for {os.path.basename(sample_path)}: {result}")
        print(f"Prediction Confidence: {prediction}")
