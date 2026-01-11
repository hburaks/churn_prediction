# 17_xgboost_threshold_tuning.py
import pandas as pd
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

print("--- XGBoost Karar Eşiği Ayarlaması ---")

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
    scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
    print(f"\nXGBoost 'scale_pos_weight' parametresi hesaplandı: {scale_pos_weight:.2f}")

    # XGBoost Modelini Tanımla ve Eğit
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        random_state=42
    )
    print("\nModel eğitiliyor...")
    model.fit(X_train, y_train)
    print("- Model başarıyla eğitildi.")

    # Test Seti Üzerinde Churn Olasılıklarını Alma
    print("\nTest seti üzerinde churn olasılıkları hesaplanıyor...")
    probabilities = model.predict_proba(X_test)[:, 1]

    # Farklı Karar Eşikleri ile Değerlendirme
    print("\n--- Farklı Karar Eşikleri ile Model Değerlendirme ---")
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

    for threshold in thresholds:
        print(f"\n--- Eşik Değeri: {threshold} ---")
        y_pred_threshold = (probabilities >= threshold).astype(int)

        print("  Karışıklık Matrisi:\n", confusion_matrix(y_test, y_pred_threshold))
        report = classification_report(y_test, y_pred_threshold, target_names=['Not Churn (0)', 'Churn (1)'], output_dict=True)
        print("  Sınıflandırma Raporu (Churn (1) Sınıfı için):")
        print(f"    Precision: {report['Churn (1)']['precision']:.4f}")
        print(f"    Recall:    {report['Churn (1)']['recall']:.4f}")
        print(f"    F1-Score:  {report['Churn (1)']['f1-score']:.4f}")
        print("-" * 30)

except FileNotFoundError as e:
    print(f"Hata: Dosya bulunamadı - {e.filename}")
except Exception as e:
    print(f"Bir hata oluştu: {e}")
