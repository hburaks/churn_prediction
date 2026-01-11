# 16_train_xgboost.py
import pandas as pd
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

print("--- XGBoost Modeli Eğitimi ---")

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

    # XGBoost için sınıf dengesizliği parametresini hesapla
    # scale_pos_weight = count(negative class) / count(positive class)
    scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
    print(f"\nXGBoost 'scale_pos_weight' parametresi hesaplandı: {scale_pos_weight:.2f}")

    # XGBoost Modelini Tanımla
    # Temel parametreler ve dengesizlik ayarı ile başlıyoruz
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False, # Etiket kodlayıcı uyarısını önlemek için
        random_state=42
    )

    # Modeli Eğitme
    print("\nModel eğitiliyor...")
    model.fit(X_train, y_train)
    print("- Model başarıyla eğitildi.")

    # Test Seti Üzerinde Tahmin Yapma ve Değerlendirme
    print("\nTest seti üzerinde tahminler yapılıyor ve değerlendiriliyor...")
    y_pred = model.predict(X_test)

    print("\n--- XGBoost Model Değerlendirme Sonuçları ---")
    print("\nKarışıklık Matrisi (Confusion Matrix):")
    print(confusion_matrix(y_test, y_pred))
    print("\nSınıflandırma Raporu (Classification Report):")
    print(classification_report(y_test, y_pred, target_names=['Not Churn (0)', 'Churn (1)']))
    
    print("\n--- Karşılaştırma ---")
    print("Lojistik Regresyon (En İyi): F1-Score: ~0.42")
    report = classification_report(y_test, y_pred, output_dict=True)
    f1 = report['1']['f1-score']
    print(f"XGBoost (İlk Model): F1-Score: {f1:.4f}")

except FileNotFoundError as e:
    print(f"Hata: Dosya bulunamadı - {e.filename}")
except Exception as e:
    print(f"Bir hata oluştu: {e}")
