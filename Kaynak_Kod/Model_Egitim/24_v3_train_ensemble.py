import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import VotingClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import time

print("--- Ensemble Learning (Topluluk Öğrenmesi) Başlatılıyor (Adım 24-V3) ---")

# Dosya yolları
current_dir = os.getcwd()
input_file_path = os.path.join(current_dir, 'model_ready_reduced_dataset_v2.csv')
results_out = os.path.join(current_dir, 'ensemble_results.md')

try:
    # 1. Veri Setini Yükle
    print(f"'{input_file_path}' yükleniyor...")
    df = pd.read_csv(input_file_path)
    X = df.drop('is_churn', axis=1)
    y = df['is_churn']

    # 2. Eğitim ve Test Ayırma
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Dengesizlik oranı (Scale Pos Weight)
    scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
    print(f"Dengesizlik Oranı (Scale Pos Weight): {scale_pos_weight:.2f}")

    # 3. Modelleri Tanımla
    print("\nModeller hazırlanıyor...")
    
    # XGBoost (Optimize Edilmiş Parametrelerle)
    clf_xgb = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight,
        learning_rate=0.2,
        max_depth=9,
        n_estimators=291,
        subsample=0.82,
        colsample_bytree=0.88,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )

    # LightGBM (Dengesiz veri için 'is_unbalance=True' veya 'scale_pos_weight')
    clf_lgbm = lgb.LGBMClassifier(
        objective='binary',
        metric='binary_logloss',
        scale_pos_weight=scale_pos_weight,
        learning_rate=0.1,
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    # CatBoost (Otomatik sınıf ağırlığı hesaplar)
    clf_cat = CatBoostClassifier(
        loss_function='Logloss',
        auto_class_weights='Balanced',
        iterations=300,
        learning_rate=0.1,
        random_seed=42,
        verbose=0 # Sessiz mod
    )

    # 4. Modelleri Teker Teker Eğit (Performanslarını görmek için)
    print("\n1. XGBoost Eğitiliyor...")
    start = time.time()
    clf_xgb.fit(X_train, y_train)
    print(f"- XGBoost Tamamlandı ({time.time() - start:.2f}s)")

    print("\n2. LightGBM Eğitiliyor...")
    start = time.time()
    clf_lgbm.fit(X_train, y_train)
    print(f"- LightGBM Tamamlandı ({time.time() - start:.2f}s)")

    print("\n3. CatBoost Eğitiliyor...")
    start = time.time()
    clf_cat.fit(X_train, y_train)
    print(f"- CatBoost Tamamlandı ({time.time() - start:.2f}s)")

    # 5. Ensemble (Voting) - Soft Voting
    # Soft Voting: Olasılıkların ortalamasını alır
    print("\n4. Ensemble (Voting) Tahmini Yapılıyor...")
    
    # Test seti üzerinde olasılıkları al
    prob_xgb = clf_xgb.predict_proba(X_test)[:, 1]
    prob_lgbm = clf_lgbm.predict_proba(X_test)[:, 1]
    prob_cat = clf_cat.predict_proba(X_test)[:, 1]
    
    # Ortalama olasılık
    avg_prob = (prob_xgb + prob_lgbm + prob_cat) / 3

    # 6. Threshold Tuning (Eşik Ayarlaması)
    print("\n--- Eşik Değeri (Threshold) Optimizasyonu ---")
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    best_f1 = 0
    best_thr = 0.5

    for thr in thresholds:
        y_pred_thr = (avg_prob >= thr).astype(int)
        rep = classification_report(y_test, y_pred_thr, output_dict=True, zero_division=0)
        f1 = rep['1']['f1-score']
        print(f"Threshold: {thr} -> F1: {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    print(f"\nEn iyi Eşik Değeri: {best_thr} (F1: {best_f1:.4f})")

    # Final Rapor
    y_pred_final = (avg_prob >= best_thr).astype(int)
    final_rep = classification_report(y_test, y_pred_final, target_names=['Not Churn (0)', 'Churn (1)'])
    
    print("\n--- Final Ensemble Model Sonuçları ---")
    print(final_rep)
    
    # Sonuçları Kaydet
    with open(results_out, 'w') as f:
        f.write("# Ensemble Model (XGB + LGBM + Cat) Sonuçları\n\n")
        f.write(f"En İyi Eşik Değeri: {best_thr}\n\n")
        f.write("## Sınıflandırma Raporu\n")
        f.write(f"```\n{final_rep}\n```\n")

except Exception as e:
    print(f"Bir hata oluştu: {e}")
