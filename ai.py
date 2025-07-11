import numpy as np
import pandas as pd
import tensorflow as tf
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

def set_random_seed(seed=42):
    """Mengatur seed untuk reproducibility"""
    tf.random.set_seed(seed)
    np.random.set_seed(seed)
    random.seed(seed)

def load_and_preprocess_data(file_path):
    """Memuat dan melakukan pra-pemrosesan data"""
    data = pd.read_csv(file_path)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data['Indonesia(IDR)'].values.reshape(-1, 1))
    return data_scaled, scaler

def create_sequences(data, seq_length):
    """Membuat urutan data untuk time series"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def prepare_data(data_scaled, seq_length=4, test_size=0.4):
    """Menyiapkan data untuk pelatihan"""
    X, y = create_sequences(data_scaled, seq_length)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return train_test_split(X, y, test_size=test_size, random_state=42)

def build_model(seq_length):
    """Membangun arsitektur model"""
    model = Sequential([
        Conv1D(filters=32, kernel_size=3, activation='relu', 
               input_shape=(seq_length, 1)),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        LSTM(50, return_sequences=False,
             kernel_regularizer=tf.keras.regularizers.l2(0.01),
             recurrent_regularizer=tf.keras.regularizers.l2(0.01)),
        Dropout(0.3),
        
        Dense(32, activation='relu',
              kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Dense(1)
    ])
    return model

def compile_and_train_model(model, X_train, y_train, X_test, y_test):
    """Mengompilasi dan melatih model"""
    optimizer = Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=16,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )
    return history

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Mengevaluasi performa model"""
    train_loss = model.evaluate(X_train, y_train, verbose=0)
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    
    print(f'RMSE Data Latih: {np.sqrt(train_loss)}')
    print(f'RMSE Data Uji: {np.sqrt(test_loss)}')

def plot_predictions(y_test, y_pred):
    """Memvisualisasikan hasil prediksi"""
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='Aktual', linewidth=2)
    plt.plot(y_pred, label='Prediksi', linewidth=2)
    plt.title('Harga Emas Aktual vs Prediksi')
    plt.xlabel('Waktu')
    plt.ylabel('Harga Ternormalisasi')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_training_history(history):
    """Memvisualisasikan history pelatihan"""
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Loss Pelatihan')
    plt.plot(history.history['val_loss'], label='Loss Validasi')
    plt.title('Loss Model Selama Pelatihan')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # Inisialisasi
    set_random_seed()
    
    # Persiapan data
    data_scaled, scaler = load_and_preprocess_data('1990-2021.csv')
    X_train, X_test, y_train, y_test = prepare_data(data_scaled)
    
    # Pembuatan dan pelatihan model
    model = build_model(seq_length=4)
    history = compile_and_train_model(model, X_train, y_train, X_test, y_test)
    
    # Evaluasi
    evaluate_model(model, X_train, y_train, X_test, y_test)
    
    # Prediksi dan visualisasi
    y_pred = model.predict(X_test)
    plot_predictions(y_test, y_pred)
    plot_training_history(history)

if __name__ == "__main__":
    main()
