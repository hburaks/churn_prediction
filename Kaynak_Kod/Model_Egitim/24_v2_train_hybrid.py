import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
import time

print("--- Hibrit (Wide & Deep) Model Eğitimi (Adım 24-V2) Başlatılıyor ---")

# Dosya yolları
current_dir = os.getcwd()
X_tabular_path = os.path.join(current_dir, 'hybrid_X_tabular.npy')
X_rnn_path = os.path.join(current_dir, 'hybrid_X_rnn.npy')
y_path = os.path.join(current_dir, 'hybrid_y.npy')
model_out = os.path.join(current_dir, 'hybrid_model_v2.keras')
results_out = os.path.join(current_dir, 'hybrid_results_v2.md')

try:
    # 1. Verileri yükle
    print("Hizalanmış veriler yükleniyor...")
    X_tab = np.load(X_tabular_path)
    X_seq = np.load(X_rnn_path)
    y = np.load(y_path)
    
    # 2. Eğitim ve Test Ayırma (İndeks üzerinden hizalamayı koruyarak)
    indices = np.arange(len(y))
    idx_train, idx_test = train_test_split(indices, test_size=0.2, random_state=42, stratify=y)
    
    X_tab_train, X_tab_test = X_tab[idx_train], X_tab[idx_test]
    X_seq_train, X_seq_test = X_seq[idx_train], X_seq[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]
    
    print(f"Eğitim seti: {X_tab_train.shape}, Test seti: {X_tab_test.shape}")

    # 3. Class Weight Hesaplama
    cw = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    cw_dict = dict(enumerate(cw))
    print(f"Sınıf Ağırlıkları: {cw_dict}")

    # 4. Hibrit Model Mimarisi
    # --- Sol Kol: Tabular (Wide) ---
    input_tab = Input(shape=(X_tab_train.shape[1],), name='tabular_input')
    tab_branch = Dense(64, activation='relu')(input_tab)
    tab_branch = BatchNormalization()(tab_branch)
    tab_branch = Dropout(0.3)(tab_branch)

    # --- Sağ Kol: Sequence (Deep) ---
    input_seq = Input(shape=(X_seq_train.shape[1], X_seq_train.shape[2]), name='sequence_input')
    seq_branch = LSTM(64, return_sequences=False)(input_seq)
    seq_branch = BatchNormalization()(seq_branch)
    seq_branch = Dropout(0.3)(seq_branch)

    # --- Birleşme (Concatenate) ---
    merged = Concatenate()([tab_branch, seq_branch])
    merged = Dense(32, activation='relu')(merged)
    merged = Dropout(0.2)(merged)
    output = Dense(1, activation='sigmoid')(merged)

    model = Model(inputs=[input_tab, input_seq], outputs=output)

    # 5. Derleme ve Eğitim
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    
    early_stop = EarlyStopping(monitor='val_auc', mode='max', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_auc', mode='max', factor=0.5, patience=2, verbose=1)

    print("\nHibrit model eğitiliyor...")
    start_time = time.time()
    model.fit(
        [X_tab_train, X_seq_train], y_train,
        epochs=30,
        batch_size=1024,
        validation_split=0.2,
        callbacks=[early_stop, reduce_lr],
        class_weight=cw_dict,
        verbose=1
    )
    end_time = time.time()
    
    model.save(model_out)
    
    # 6. Değerlendirme ve Threshold Tuning
    print("\n--- Eşik Değeri (Threshold) Optimizasyonu ---")
    y_pred_prob = model.predict([X_tab_test, X_seq_test])
    
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    best_f1 = 0
    best_thr = 0.5

    for thr in thresholds:
        y_pred_thr = (y_pred_prob >= thr).astype(int)
        rep = classification_report(y_test, y_pred_thr, output_dict=True, zero_division=0)
        f1 = rep['1']['f1-score']
        print(f"Threshold: {thr} -> F1: {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    # En iyi sonuçla final raporu
    y_pred_final = (y_pred_prob >= best_thr).astype(int)
    final_rep = classification_report(y_test, y_pred_final, target_names=['Not Churn (0)', 'Churn (1)'])
    print("\n--- Final Hibrit Model Sonuçları ---")
    print(final_rep)

    with open(results_out, 'w') as f:
        f.write("# Hibrit Model (V2) Sonuçları\n\n")
        f.write(f"Eğitim Süresi: {end_time - start_time:.2f} saniye\n")
        f.write(f"En İyi Eşik: {best_thr}\n\n")
        f.write("## Sınıflandırma Raporu\n")
        f.write(f"```\n{final_rep}\n```\n")

except Exception as e:
    print(f"Bir hata oluştu: {e}")
