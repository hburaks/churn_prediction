# 09_train_logreg_feature_selection.py
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings

# Olası convergence uyarılarını bastır
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

print("--- Özellik Seçimi Sonrası Lojistik Regresyon Eğitimi ---")

# Dosya yolunu tanımla
current_dir = os.getcwd()
input_file_path = os.path.join(current_dir, 'model_ready_dataset.csv')

try:
    # Modellenmeye hazır veri setini yükle
    print(f"'{input_file_path}' yükleniyor...")
    df = pd.read_csv(input_file_path)
    print(f"Orijinal DataFrame boyutu: {df.shape}")

    # 1. Özellik Seçimi: Etkisiz sütunları kaldırma
    print("\n1. Etkisiz özellikler (city_* ve gender_*) kaldırılıyor...")
    
    # Kaldırılacak sütunları bul
    cols_to_drop = [col for col in df.columns if col.startswith('city_') or col.startswith('gender_')]
    
    df_reduced = df.drop(columns=cols_to_drop)
    print(f"- {len(cols_to_drop)} adet sütun kaldırıldı.")
    print(f"Yeni DataFrame boyutu: {df_reduced.shape}")

    # 2. Özellik (X) ve Hedef (y) Değişkenlerini Ayırma
    print("\n2. Özellik (X) ve Hedef (y) değişkenleri ayrılıyor...")
    X = df_reduced.drop('is_churn', axis=1)
    y = df_reduced['is_churn']
    print(f"Özellik (X) boyutu: {X.shape}")
    print(f"Hedef (y) boyutu: {y.shape}")

    # 3. Veri Setini Eğitim ve Test Olarak Ayırma
    print("\n3. Veri seti eğitim ve test olarak ayrılıyor...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 4. Özellikleri Ölçeklendirme
    print("\n4. Özellikler ölçeklendiriliyor...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Lojistik Regresyon Modelini Eğitme
    print("\n5. Lojistik Regresyon modeli (yeni özellik setiyle) eğitiliyor...")
    model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    print("- Model başarıyla eğitildi.")

    # 6. Test Seti Üzerinde Tahmin Yapma ve Değerlendirme
    print("\n6. Test seti üzerinde tahminler yapılıyor ve değerlendiriliyor...")
    y_pred = model.predict(X_test_scaled)

    print("\n--- Yeni Model Değerlendirme Sonuçları (Özellik Seçimi Sonrası) ---")
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nDoğruluk (Accuracy): {accuracy:.4f}")

    print("\nKarışıklık Matrisi (Confusion Matrix):")
    print(confusion_matrix(y_test, y_pred))

    print("\nSınıflandırma Raporu (Classification Report):")
    print(classification_report(y_test, y_pred, target_names=['Not Churn (0)', 'Churn (1)']))
    
    print("-" * 50)
    print("Karşılaştırma:")
    print("Önceki Model (Tüm Özellikler) -> Churn (1) Precision: 0.27, Recall: 0.88, F1-Score: 0.41")
    new_report = classification_report(y_test, y_pred, output_dict=True)
    new_precision = new_report['Churn (1)']['precision']
    new_recall = new_report['Churn (1)']['recall']
    new_f1 = new_report['Churn (1)']['f1-score']
    print(f"Yeni Model (Seçilmiş Özellikler) -> Churn (1) Precision: {new_precision:.2f}, Recall: {new_recall:.2f}, F1-Score: {new_f1:.2f}")


except FileNotFoundError as e:
    print(f"Hata: Dosya bulunamadı - {e.filename}. Lütfen dosya yollarını kontrol edin.")
except Exception as e:
    print(f"Bir hata oluştu: {e}")
