import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib
import librosa
import parselmouth
from parselmouth.praat import call
from keras.utils import to_categorical
from keras.optimizers import Nadam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, Dropout, Dense, MaxPooling1D, GlobalAveragePooling1D, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
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




def plot_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    
    plt.show()

def normalize_data(X, scaler=None):
    X_reshaped = X.reshape(X.shape[0], -1)
    if scaler is None:
        scaler = StandardScaler().fit(X_reshaped)
    X_scaled = scaler.transform(X_reshaped)
    return X_scaled.reshape(X.shape[0], X.shape[1], 1), scaler

def augment_data(X, Y):
    augmented_X, augmented_Y = [], []
    for x, y in zip(X, Y):
        # Menambahkan original
        augmented_X.append(x)
        augmented_Y.append(y)

        # Menambahkan noise
        noise = np.random.normal(0, 0.005, x.shape)
        augmented_X.append(x + noise)
        augmented_Y.append(y)

        # Pitch shift
        pitch_shift = np.random.randint(-3, 3)
        pitch_shifted = librosa.effects.pitch_shift(x.flatten(), sr=22050, n_steps=pitch_shift)
        augmented_X.append(pitch_shifted.reshape(-1, 1))
        augmented_Y.append(y)

        # Time stretching
        stretch_factor = np.random.uniform(0.8, 1.2)
        try:
            stretched = librosa.effects.time_stretch(x.flatten(), rate=stretch_factor)
            if len(stretched) < x.size:
                # Pad if stretched file is shorter
                stretched = np.pad(stretched, (0, x.size - len(stretched)), 'constant')
            elif len(stretched) > x.size:
                # Truncate if stretched file is longer
                stretched = stretched[:x.size]
            augmented_X.append(stretched.reshape(-1, 1))
            augmented_Y.append(y)
        except ValueError:
            # Skip augmentation if the audio is too short after stretching
            continue

        # Dynamic range compression
        compressed = librosa.effects.percussive(x.flatten(), margin=8)
        augmented_X.append(compressed.reshape(-1, 1))
        augmented_Y.append(y)

        # Random gain
        gain_change = np.random.uniform(0.9, 1.1)
        gained = x.flatten() * gain_change
        augmented_X.append(gained.reshape(-1, 1))
        augmented_Y.append(y)

    return np.array(augmented_X), np.array(augmented_Y)



def build_enhanced_model(input_shape):
    model = Sequential([
        Conv1D(128, padding='same', kernel_size=3, input_shape=input_shape),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        MaxPooling1D(pool_size=2, strides=1),
        
        Conv1D(256, padding='same', kernel_size=3),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        MaxPooling1D(pool_size=2, strides=1),
        
        Conv1D(512, padding='same', kernel_size=3),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        MaxPooling1D(pool_size=2, strides=1),
        
        GlobalAveragePooling1D(),
        Dense(1024, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        LeakyReLU(alpha=0.1),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])
    model.summary()
    optimizer = Nadam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

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

def process_and_train(X, Y, epochs=250):
    le = LabelEncoder()
    dataset_y_encoded = le.fit_transform(Y)
    dataset_y_onehot = to_categorical(dataset_y_encoded)
    X, scaler = normalize_data(X)
    
    # Simpan scaler
    joblib.dump(scaler, 'tesfix.pkl')
    
    X_augmented, Y_augmented = augment_data(X, dataset_y_onehot)
    
    X_train, X_val, Y_train, Y_val = train_test_split(X_augmented, Y_augmented, test_size=0.2, random_state=42, stratify=Y_augmented)
    
    model = build_enhanced_model(X_train.shape[1:])
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.00001)
    model_checkpoint = ModelCheckpoint("tesfix.h5", monitor='val_accuracy', save_best_only=True)
    
    history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=epochs, batch_size=64, callbacks=[early_stopping, reduce_lr, model_checkpoint])
    return model, history

if __name__ == '__main__':
    healthy_folder = "Z:/speech depressed/NonDepression"
    depressed_folder = "Z:/speech depressed/depresidata"
    X, Y, healthy_paths, depressed_paths = load_data(healthy_folder, depressed_folder)
    trained_model, history = process_and_train(X, Y)

    Y_onehot = to_categorical(Y)
    X, _ = normalize_data(X)

    plot_history(history)
    evaluation = trained_model.evaluate(X, Y_onehot)
    print(f"Final Loss: {evaluation[0]}")
    print(f"Final Accuracy: {evaluation[1]}")

    best_model = tf.keras.models.load_model("tesfix.h5")
    best_evaluation = best_model.evaluate(X, Y_onehot)
    print(f"Best Model Final Loss: {best_evaluation[0]}")
    print(f"Best Model Final Accuracy: {best_evaluation[1]}")
