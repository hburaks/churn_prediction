# 10_logreg_threshold_tuning.py
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
import numpy as np

# Olası convergence uyarılarını bastır
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

print("--- Lojistik Regresyon Karar Eşiği Ayarlaması ---")

# Dosya yolunu tanımla
current_dir = os.getcwd()
input_file_path = os.path.join(current_dir, 'model_ready_reduced_dataset.csv')

try:
    # Azaltılmış veri setini yükle
    print(f"'{input_file_path}' yükleniyor...")
    df = pd.read_csv(input_file_path)
    print(f"Veri seti boyutu: {df.shape}")

    # 1. Özellik (X) ve Hedef (y) Değişkenlerini Ayırma
    print("\n1. Özellik (X) ve Hedef (y) değişkenleri ayrılıyor...")
    X = df.drop('is_churn', axis=1)
    y = df['is_churn']

    # 2. Veri Setini Eğitim ve Test Olarak Ayırma
    print("\n2. Veri seti eğitim ve test olarak ayrılıyor...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. Özellikleri Ölçeklendirme
    print("\n3. Özellikler ölçeklendiriliyor...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Lojistik Regresyon Modelini Eğitme
    print("\n4. Lojistik Regresyon modeli eğitiliyor...")
    model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    print("- Model başarıyla eğitildi.")

    # 5. Test Seti Üzerinde Churn Olasılıklarını Alma
    print("\n5. Test seti üzerinde churn olasılıkları hesaplanıyor...")
    # predict_proba, her sınıf için olasılıkları verir. Biz Churn (1) sınıfının olasılığını alıyoruz.
    probabilities = model.predict_proba(X_test_scaled)[:, 1]

    # 6. Farklı Karar Eşikleri ile Değerlendirme
    print("\n--- Farklı Karar Eşikleri ile Model Değerlendirme ---")
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

    for threshold in thresholds:
        print(f"\n--- Eşik Değeri: {threshold} ---")
        # Yeni tahminleri eşik değerine göre oluştur
        y_pred_threshold = (probabilities >= threshold).astype(int)

        # Doğruluk (Accuracy)
        accuracy = accuracy_score(y_test, y_pred_threshold)
        print(f"Doğruluk (Accuracy): {accuracy:.4f}")

        # Karışıklık Matrisi (Confusion Matrix)
        print("Karışıklık Matrisi (Confusion Matrix):")
        print(confusion_matrix(y_test, y_pred_threshold))

        # Sınıflandırma Raporu (Classification Report)
        report = classification_report(y_test, y_pred_threshold, target_names=['Not Churn (0)', 'Churn (1)'], output_dict=True)
        print("DEBUG: Raporun anahtarları:", report.keys()) # Hata ayıklama için anahtarları yazdır
        print("Sınıflandırma Raporu (Churn (1) Sınıfı için):")
        print(f"  Precision: {report['Churn (1)']['precision']:.4f}")
        print(f"  Recall:    {report['Churn (1)']['recall']:.4f}")
        print(f"  F1-Score:  {report['Churn (1)']['f1-score']:.4f}")
        print("-" * 30)

except FileNotFoundError as e:
    print(f"Hata: Dosya bulunamadı - {e.filename}. Lütfen dosya yollarını kontrol edin.")
except Exception as e:
    print(f"Bir hata oluştu: {e}")
