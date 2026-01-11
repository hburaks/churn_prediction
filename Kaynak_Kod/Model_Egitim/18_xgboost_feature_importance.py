# 18_xgboost_feature_importance.py
import pandas as pd
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

print("--- XGBoost Modelinin Özellik Önemini Analiz Etme ---")

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
    print("\nModel eğitiliyor (özellik önemini hesaplamak için)...")
    model.fit(X_train, y_train)
    print("- Model başarıyla eğitildi.")

    # Özellik önemini al
    feature_importances = model.feature_importances_

    # Özellik isimleri ile önem skorlarını birleştir
    feature_importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': feature_importances
    })

    # Önem sırasına göre sırala
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

    print("\n--- En Önemli 20 Özellik (XGBoost) ---")
    print(feature_importance_df.head(20).to_string(index=False))

    # İsteğe bağlı: Özellik önemini görselleştirme
    # plt.figure(figsize=(10, 8))
    # xgb.plot_importance(model, max_num_features=20, importance_type='weight')
    # plt.title('XGBoost Feature Importance (Weight)')
    # plt.show()

except FileNotFoundError as e:
    print(f"Hata: Dosya bulunamadı - {e.filename}")
except Exception as e:
    print(f"Bir hata oluştu: {e}")
