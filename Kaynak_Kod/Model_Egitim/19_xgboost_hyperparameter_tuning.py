# 19_xgboost_hyperparameter_tuning.py
import pandas as pd
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import uniform, randint
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

print("--- XGBoost Hiperparametre Optimizasyonu (RandomizedSearchCV) ---")

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

    # XGBoost Modelini Tanımla (başlangıç parametreleri)
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False, # Etiket kodlayıcı uyarısını önlemek için
        random_state=42,
        n_jobs=-1 # Tüm CPU çekirdeklerini kullan
    )

    # Ayarlanacak hiperparametreler için dağılımlar
    param_distributions = {
        'n_estimators': randint(100, 500),  # Ağaç sayısı
        'learning_rate': uniform(0.01, 0.2), # Öğrenme oranı
        'max_depth': randint(3, 10),       # Ağaç derinliği
        'subsample': uniform(0.6, 0.4),    # Her ağaç için örnek oranı
        'colsample_bytree': uniform(0.6, 0.4), # Her ağaç için özellik oranı
        'gamma': uniform(0, 0.2),          # Minimum kayıp azaltma
        'lambda': uniform(1, 2),           # L2 regülarizasyon
        'alpha': uniform(0, 0.2)           # L1 regülarizasyon
    }

    # RandomizedSearchCV kurulumu
    # n_iter: Denenecek parametre kombinasyonu sayısı
    # cv: Çapraz doğrulama kat sayısı
    # scoring: Optimizasyon metriği (F1-score, churn sınıfı için)
    print("\nRandomizedSearchCV başlatılıyor...")
    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_distributions,
        n_iter=50, # Denenecek kombinasyon sayısı (daha fazla zaman alabilir)
        scoring='f1', # F1-score'u optimize et
        cv=3,         # 3 katlı çapraz doğrulama
        verbose=1,
        random_state=42,
        n_jobs=-1     # Tüm CPU çekirdeklerini kullan
    )

    # Optimizasyonu başlat
    random_search.fit(X_train, y_train)

    print("\n--- Hiperparametre Optimizasyonu Tamamlandı ---")
    print("En iyi F1-Skoru:", random_search.best_score_)
    print("En iyi parametreler:", random_search.best_params_)

    # En iyi modeli al
    best_model = random_search.best_estimator_

    # Test seti üzerinde en iyi modelin performansını değerlendir
    print("\nEn iyi modelin test seti üzerindeki performansı değerlendiriliyor...")
    y_pred = best_model.predict(X_test)

    print("\n--- En İyi XGBoost Model Değerlendirme Sonuçları ---")
    print("\nKarışıklık Matrisi (Confusion Matrix):")
    print(confusion_matrix(y_test, y_pred))
    print("\nSınıflandırma Raporu (Classification Report):")
    print(classification_report(y_test, y_pred, target_names=['Not Churn (0)', 'Churn (1)']))
    
    print("\n--- Karşılaştırma ---")
    print("XGBoost (İlk Model): F1-Score: 0.4863")
    report = classification_report(y_test, y_pred, output_dict=True)
    f1 = report['1']['f1-score']
    print(f"XGBoost (Optimize Edilmiş Model): F1-Score: {f1:.4f}")

except FileNotFoundError as e:
    print(f"Hata: Dosya bulunamadı - {e.filename}")
except Exception as e:
    print(f"Bir hata oluştu: {e}")
