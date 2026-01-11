# 07_train_logistic_regression.py
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings

# Olası convergence uyarılarını bastır
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

print("--- Lojistik Regresyon Modeli Eğitimi ve Değerlendirmesi ---")

# Dosya yolunu tanımla
current_dir = os.getcwd()
input_file_path = os.path.join(current_dir, 'model_ready_dataset.csv')

try:
    # Modellenmeye hazır veri setini yükle
    print(f"'{input_file_path}' yükleniyor...")
    df = pd.read_csv(input_file_path)
    print(f"Veri seti boyutu: {df.shape}")

    # 1. Özellik (X) ve Hedef (y) Değişkenlerini Ayırma
    print("\n1. Özellik (X) ve Hedef (y) değişkenleri ayrılıyor...")
    X = df.drop('is_churn', axis=1)
    y = df['is_churn']
    print(f"Özellik (X) boyutu: {X.shape}")
    print(f"Hedef (y) boyutu: {y.shape}")

    # 2. Veri Setini Eğitim ve Test Olarak Ayırma
    # stratify=y, eğitim ve test setlerindeki 'is_churn' oranının orijinal veri setiyle aynı olmasını sağlar.
    # Bu, dengesiz veri setleri için çok önemlidir.
    print("\n2. Veri seti eğitim ve test olarak ayrılıyor...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Eğitim seti boyutu: {X_train.shape}")
    print(f"Test seti boyutu: {X_test.shape}")

    # 3. Özellikleri Ölçeklendirme (Feature Scaling)
    # Lojistik Regresyon gibi birçok model, özelliklerin aynı ölçekte olmasından faydalanır.
    print("\n3. Özellikler ölçeklendiriliyor (StandardScaler)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("- Özellikler başarıyla ölçeklendirildi.")

    # 4. Lojistik Regresyon Modelini Eğitme
    # class_weight='balanced', dengesiz veri setindeki azınlık sınıfına (churn=1) daha fazla önem verir.
    print("\n4. Lojistik Regresyon modeli eğitiliyor...")
    model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    print("- Model başarıyla eğitildi.")

    # 5. Test Seti Üzerinde Tahmin Yapma
    print("\n5. Test seti üzerinde tahminler yapılıyor...")
    y_pred = model.predict(X_test_scaled)

    # 6. Model Performansını Değerlendirme
    print("\n--- Model Değerlendirme Sonuçları ---")
    
    # Doğruluk (Accuracy)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nDoğruluk (Accuracy): {accuracy:.4f}")

    # Karışıklık Matrisi (Confusion Matrix)
    # TN | FP
    # FN | TP
    print("\nKarışıklık Matrisi (Confusion Matrix):")
    print(confusion_matrix(y_test, y_pred))

    # Sınıflandırma Raporu (Classification Report)
    print("\nSınıflandırma Raporu (Classification Report):")
    print(classification_report(y_test, y_pred, target_names=['Not Churn (0)', 'Churn (1)']))
    print("-" * 50)
    print("Raporun Yorumu:")
    print("- precision (kesinlik): Modelin 'Churn' olarak tahmin ettiği kullanıcıların ne kadarının gerçekten 'Churn' olduğunu gösterir.")
    print("- recall (duyarlılık): Gerçekten 'Churn' olan kullanıcıların ne kadarını modelin doğru tespit edebildiğini gösterir.")
    print("- f1-score: Precision ve recall metriklerinin harmonik ortalamasıdır, dengesiz veri setleri için iyi bir ölçüttür.")


except FileNotFoundError as e:
    print(f"Hata: Dosya bulunamadı - {e.filename}. Lütfen dosya yollarını kontrol edin.")
except Exception as e:
    print(f"Bir hata oluştu: {e}")
