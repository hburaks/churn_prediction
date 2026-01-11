import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import time

print("--- LSTM Model Eğitimi Başlatılıyor ---")

# Dosya yolları
current_dir = os.getcwd()
X_rnn_path = os.path.join(current_dir, 'rnn_X_sequences.npz')
y_rnn_path = os.path.join(current_dir, 'rnn_y_targets.npy')
model_out = os.path.join(current_dir, 'lstm_model.keras')
results_out = os.path.join(current_dir, 'lstm_results.md')

try:
    # 1. Verileri yükle
    print("Sekans verileri yükleniyor...")
    with np.load(X_rnn_path) as data:
        X = data['data']
    y = np.load(y_rnn_path)
    print(f"X boyutu: {X.shape}, y boyutu: {y.shape}")

    # 2. Eğitim ve Test Ayırma
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Eğitim seti: {X_train.shape}, Test seti: {X_test.shape}")

    # 3. Model Mimarisi
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    
    # 4. Eğitim
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    print("\nLSTM modeli eğitiliyor...")
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=20, # LSTM daha yavaştır, epoch sayısını düşük tutalım
        batch_size=512,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )
    end_time = time.time()
    
    # 5. Kaydet ve Değerlendir
    model.save(model_out)
    print(f"\nModel kaydedildi: {model_out}")

    print("\nTest seti üzerinde tahminler yapılıyor...")
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob >= 0.5).astype(int)

    report = classification_report(y_test, y_pred, target_names=['Not Churn (0)', 'Churn (1)'])
    cm = confusion_matrix(y_test, y_pred)

    print("\n--- LSTM Model Sonuçları ---")
    print(report)

    with open(results_out, 'w') as f:
        f.write("# LSTM Model Sonuçları\n\n")
        f.write(f"Eğitim Süresi: {end_time - start_time:.2f} saniye\n\n")
        f.write("## Sınıflandırma Raporu\n")
        f.write(f"```\n{report}\n```\n\n")
        f.write("## Karışıklık Matrisi\n")
        f.write(f"```\n{cm}\n```\n")

except Exception as e:
    print(f"Bir hata oluştu: {e}")
