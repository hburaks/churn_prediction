# 15_logreg_final_tuning.py
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

print("--- Lojistik Regresyon Nihai Ayarlama (Eşik Ayarı ve Undersampling) ---")

# Dosya yolunu tanımla
current_dir = os.getcwd()
input_file_path = os.path.join(current_dir, 'model_ready_reduced_dataset_v2.csv')

try:
    # Veri setini yükle
    print(f"'{input_file_path}' yükleniyor...")
    df = pd.read_csv(input_file_path)
    print(f"DataFrame boyutu: {df.shape}")

    # Özellik (X) ve Hedef (y) Değişkenlerini Ayırma
    X = df.drop('is_churn', axis=1)
    y = df['is_churn']

    # Veri Setini Eğitim ve Test Olarak Ayırma
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Özellikleri Ölçeklendirme
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- BÖLÜM 1: STANDART MODEL İLE EŞİK AYARLAMASI ---
    print("\n" + "="*50)
    print("BÖLÜM 1: Standart Model ile Eşik Ayarlaması")
    print("="*50)

    model_standard = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    model_standard.fit(X_train_scaled, y_train)
    
    print("\nStandart Model (class_weight='balanced') Sonuçları:")
    probabilities_standard = model_standard.predict_proba(X_test_scaled)[:, 1]
    
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    for threshold in thresholds:
        print(f"\n--- Eşik Değeri: {threshold} ---")
        y_pred_threshold = (probabilities_standard >= threshold).astype(int)
        report = classification_report(y_test, y_pred_threshold, output_dict=True)
        print("  Karışıklık Matrisi:\n", confusion_matrix(y_test, y_pred_threshold))
        print(f"  Precision (Churn): {report['1']['precision']:.4f}")
        print(f"  Recall (Churn):    {report['1']['recall']:.4f}")
        print(f"  F1-Score (Churn):  {report['1']['f1-score']:.4f}")

    # --- BÖLÜM 2: UNDERSAMPLING İLE MODEL EĞİTİMİ VE EŞİK AYARLAMASI ---
    print("\n" + "="*50)
    print("BÖLÜM 2: Undersampling ile Model Eğitimi ve Eşik Ayarlaması")
    print("="*50)

    # Undersampling objesini oluştur
    rus = RandomUnderSampler(random_state=42)
    print("\nEğitim verisi üzerinde Undersampling uygulanıyor...")
    X_train_resampled, y_train_resampled = rus.fit_resample(X_train_scaled, y_train)
    print("Orijinal eğitim verisi dağılımı:", y_train.value_counts().to_dict())
    print("Dengelenmiş eğitim verisi dağılımı:", y_train_resampled.value_counts().to_dict())

    # Yeni modeli dengelenmiş veri üzerinde eğit (class_weight'e gerek yok artık)
    print("\nUndersampling yapılmış veri üzerinde yeni model eğitiliyor...")
    model_resampled = LogisticRegression(random_state=42, max_iter=1000)
    model_resampled.fit(X_train_resampled, y_train_resampled)

    print("\nUndersampled Model Sonuçları (Orijinal Test Seti Üzerinde):")
    probabilities_resampled = model_resampled.predict_proba(X_test_scaled)[:, 1]

    for threshold in thresholds:
        print(f"\n--- Eşik Değeri: {threshold} ---")
        y_pred_threshold = (probabilities_resampled >= threshold).astype(int)
        report = classification_report(y_test, y_pred_threshold, output_dict=True)
        print("  Karışıklık Matrisi:\n", confusion_matrix(y_test, y_pred_threshold))
        print(f"  Precision (Churn): {report['1']['precision']:.4f}")
        print(f"  Recall (Churn):    {report['1']['recall']:.4f}")
        print(f"  F1-Score (Churn):  {report['1']['f1-score']:.4f}")

except FileNotFoundError as e:
    print(f"Hata: Dosya bulunamadı - {e.filename}")
except Exception as e:
    print(f"Bir hata oluştu: {e}")
