import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
import time

print("--- LSTM Model Eğitimi (İyileştirilmiş Versiyon) Başlatılıyor ---")

# Dosya yolları
current_dir = os.getcwd()
X_rnn_path = os.path.join(current_dir, 'rnn_X_sequences.npz')
y_rnn_path = os.path.join(current_dir, 'rnn_y_targets.npy')
model_out = os.path.join(current_dir, 'lstm_model_improved.keras')
results_out = os.path.join(current_dir, 'lstm_results_improved.md')

try:
    # 1. Verileri yükle
    print("Sekans verileri yükleniyor...")
    with np.load(X_rnn_path) as data:
        X = data['data']
    y = np.load(y_rnn_path)
    
    # 2. Eğitim ve Test Ayırma
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Eğitim seti: {X_train.shape}, Test seti: {X_test.shape}")

    # 3. Class Weight Hesaplama (Dengesizlik Çözümü)
    # 0 sınıfı çok, 1 sınıfı az. 1 sınıfına daha yüksek ağırlık veriyoruz.
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))
    print(f"Sınıf Ağırlıkları: {class_weight_dict}")

    # 4. İyileştirilmiş Model Mimarisi
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        # Batch Normalization veriyi normalize ederek eğitimi stabilize eder
        BatchNormalization(), 
        LSTM(128, return_sequences=False), # Nöron sayısını artırdık
        Dropout(0.4), # Dropout oranını artırdık
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    # Adam optimizer learning rate'ini biraz düşürdük (daha hassas öğrenme)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    
    # 5. Callbacks
    early_stop = EarlyStopping(monitor='val_auc', mode='max', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_auc', mode='max', factor=0.5, patience=2, min_lr=0.00001, verbose=1)

    print("\nLSTM modeli eğitiliyor (Class Weights Aktif)...")
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=1024, # Batch size artırdık, eğitim hızlansın
        validation_split=0.2,
        callbacks=[early_stop, reduce_lr],
        class_weight=class_weight_dict, # Kritik parametre burada!
        verbose=1
    )
    end_time = time.time()
    
    # 6. Kaydet
    model.save(model_out)
    print(f"\nModel kaydedildi: {model_out}")

    # 7. Eşik Ayarlaması (Threshold Tuning)
    print("\n--- Eşik Değeri (Threshold) Optimizasyonu ---")
    y_pred_prob = model.predict(X_test)
    
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    best_f1 = 0
    best_threshold = 0.5

    for thr in thresholds:
        y_pred_thr = (y_pred_prob >= thr).astype(int)
        report = classification_report(y_test, y_pred_thr, output_dict=True, zero_division=0)
        f1 = report['1']['f1-score']
        print(f"Threshold: {thr} -> F1-Score (Churn): {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thr

    print(f"\nEn iyi Eşik Değeri: {best_threshold} (F1: {best_f1:.4f})")

    # En iyi sonuçla raporlama
    y_pred_final = (y_pred_prob >= best_threshold).astype(int)
    final_report = classification_report(y_test, y_pred_final, target_names=['Not Churn (0)', 'Churn (1)'])
    cm = confusion_matrix(y_test, y_pred_final)
    
    print("\n--- İyileştirilmiş LSTM Sonuçları ---")
    print(final_report)

    with open(results_out, 'w') as f:
        f.write("# İyileştirilmiş LSTM Model Sonuçları\n\n")
        f.write(f"Eğitim Süresi: {end_time - start_time:.2f} saniye\n")
        f.write(f"En İyi Eşik Değeri: {best_threshold}\n\n")
        f.write("## Sınıflandırma Raporu\n")
        f.write(f"```\n{final_report}\n```\n\n")

except Exception as e:
    print(f"Bir hata oluştu: {e}")
