# 12_train_logreg_v2.py
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

print("--- Yeni Özellik Setiyle Lojistik Regresyon Eğitimi (v2) ---")

# Dosya yollarını tanımla
current_dir = os.getcwd()
input_file_path = os.path.join(current_dir, 'model_ready_dataset.csv')
model_output_path = os.path.join(current_dir, 'logreg_model_v2.pkl')
scaler_output_path = os.path.join(current_dir, 'scaler_v2.pkl')

try:
    # Modellenmeye hazır yeni veri setini yükle
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

    # Lojistik Regresyon Modelini Eğitme
    print("\nModel eğitiliyor...")
    model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    print("- Model başarıyla eğitildi.")

    # Eğitilmiş modeli ve ölçekleyiciyi kaydet
    print("\nModel ve ölçekleyici diske kaydediliyor...")
    with open(model_output_path, 'wb') as f:
        pickle.dump(model, f)
    with open(scaler_output_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"- Model '{model_output_path}' olarak kaydedildi.")
    print(f"- Ölçekleyici '{scaler_output_path}' olarak kaydedildi.")

    # Test Seti Üzerinde Tahmin Yapma ve Değerlendirme
    print("\nTest seti üzerinde tahminler yapılıyor ve değerlendiriliyor...")
    y_pred = model.predict(X_test_scaled)

    print("\n--- Yeni Model Değerlendirme Sonuçları (Tüm Yeni Özellikler) ---")
    print("\nKarışıklık Matrisi (Confusion Matrix):")
    print(confusion_matrix(y_test, y_pred))
    print("\nSınıflandırma Raporu (Classification Report):")
    print(classification_report(y_test, y_pred, target_names=['Not Churn (0)', 'Churn (1)']))

except FileNotFoundError as e:
    print(f"Hata: Dosya bulunamadı - {e.filename}")
except Exception as e:
    print(f"Bir hata oluştu: {e}")
