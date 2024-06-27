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
import tkinter as tk
from tkinter import filedialog, messagebox
import sounddevice as sd
import scipy.io.wavfile as wav
import tempfile
import threading

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

class Page:
    def __init__(self, master, app):
        self.frame = tk.Frame(master, bg="#4a4a4a")
        self.app = app

    def show(self):
        self.frame.pack(fill="both", expand=True)

    def hide(self):
        self.frame.pack_forget()

class StartPage(Page):
    def __init__(self, master, app):
        super().__init__(master, app)
        self.create_widgets()

    def create_widgets(self):
        label = tk.Label(self.frame, text="Select an audio file or record to predict depression.", bg="#4a4a4a", fg="#ffffff", font=('Arial', 12))
        label.pack(pady=20)

        file_button = tk.Button(self.frame, text="Select Audio File", command=self.open_file, bg="#666666", fg="#ffffff", height=2, width=20, font=('Arial', 12))
        file_button.pack(pady=10)

        record_button = tk.Button(self.frame, text="Record from Microphone", command=self.record_audio, bg="#666666", fg="#ffffff", height=2, width=20, font=('Arial', 12))
        record_button.pack(pady=10)

    def open_file(self):
        audio_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3")])
        if audio_path:
            threading.Thread(target=self.app.show_result_page, args=(audio_path,)).start()

    def record_audio(self):
        duration = 5  # seconds
        messagebox.showinfo("Recording", f"Recording will start after you close this message and will last for {duration} seconds.")
        myrecording = sd.rec(int(duration * 44100), samplerate=44100, channels=2, dtype='float64')
        sd.wait()  # Wait until recording is finished
        temp_path = tempfile.mktemp(suffix=".wav")
        wav.write(temp_path, 44100, myrecording)
        threading.Thread(target=self.app.show_result_page, args=(temp_path,)).start()

class ResultPage(Page):
    def __init__(self, master, app):
        super().__init__(master, app)
        self.label = tk.Label(self.frame, bg="#4a4a4a", fg="#ffffff", font=('Arial', 12))
        self.label.pack(pady=20)

    def display_result(self, audio_path):
        try:
            model_path = "tesfix.h5"
            scaler_path = "tesfix.pkl"
            print(f"Processing audio file: {audio_path}")  # Debugging output
            predicted_label, prediction = predict_new_audio(audio_path, model_path, scaler_path)
            label_map = {0: "Non Depressed", 1: "Depressed"}
            result_text = f"Predicted Label: {label_map[predicted_label[0]]}\nPrediction Confidence: {prediction[0][predicted_label[0]]:.2f}%"
            self.label.config(text=result_text)
            print(f"Result: {result_text}")  # Debugging output
            if os.path.exists(audio_path) and tempfile.gettempdir() in audio_path:
                os.remove(audio_path)  # Remove temporary file if it was a recording
        except Exception as e:
            messagebox.showerror("Error", str(e))
            print(f"Error: {str(e)}")  # Debugging output

class DepressionPredictionApp:
    def __init__(self, master):
        self.master = master
        master.title("Depression Voice Prediction")
        master.geometry("800x600")
        master.configure(bg="#4a4a4a")

        self.start_page = StartPage(master, self)
        self.result_page = ResultPage(master, self)

        self.start_page.show()

    def show_result_page(self, audio_path):
        self.start_page.hide()
        self.result_page.show()
        self.result_page.display_result(audio_path)

if __name__ == "__main__":
    root = tk.Tk()
    app = DepressionPredictionApp(root)
    root.mainloop()

