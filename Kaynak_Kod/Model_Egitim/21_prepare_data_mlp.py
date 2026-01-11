import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

print("--- MLP İçin Veri Hazırlığı Başlatılıyor ---")

# Dosya yolları
current_dir = os.getcwd()
input_file_path = os.path.join(current_dir, 'model_ready_reduced_dataset_v2.csv')
X_train_out = os.path.join(current_dir, 'mlp_X_train.npy')
X_test_out = os.path.join(current_dir, 'mlp_X_test.npy')
y_train_out = os.path.join(current_dir, 'mlp_y_train.npy')
y_test_out = os.path.join(current_dir, 'mlp_y_test.npy')
scaler_out = os.path.join(current_dir, 'mlp_scaler.pkl')

try:
    # Veriyi yükle
    print(f"'{input_file_path}' yükleniyor... (Bu işlem dosya boyutu nedeniyle biraz zaman alabilir)")
    df = pd.read_csv(input_file_path)
    print(f"Veri seti boyutu: {df.shape}")

    # X ve y ayırma
    X = df.drop('is_churn', axis=1)
    y = df['is_churn']

    # Eğitim ve test ayırma (Önceki modellerle tutarlılık için aynı random_state)
    print("Veri seti eğitim ve test olarak ayrılıyor...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Ölçeklendirme (Scaling)
    print("Özellikler ölçeklendiriliyor (StandardScaler)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Verileri kaydet
    print("İşlenmiş veriler kaydediliyor...")
    np.save(X_train_out, X_train_scaled)
    np.save(X_test_out, X_test_scaled)
    np.save(y_train_out, y_train.values)
    np.save(y_test_out, y_test.values)
    
    # Scaler'ı kaydet (Gelecekte tahmin yaparken lazım olacak)
    joblib.dump(scaler, scaler_out)

    print("\n--- Hazırlık Tamamlandı ---")
    print(f"X_train: {X_train_scaled.shape}")
    print(f"X_test: {X_test_scaled.shape}")
    print(f"Dosyalar oluşturuldu: mlp_X_train.npy, mlp_X_test.npy, mlp_y_train.npy, mlp_y_test.npy, mlp_scaler.pkl")

except Exception as e:
    print(f"Bir hata oluştu: {e}")
