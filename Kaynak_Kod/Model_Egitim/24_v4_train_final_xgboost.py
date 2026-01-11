import pandas as pd
import numpy as np
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import time

print("--- Final XGBoost Eğitimi (Adım 24-V4: İleri Seviye Özellikler) ---")

# Dosya yolları
current_dir = os.getcwd()
input_file_path = os.path.join(current_dir, 'model_ready_v4_advanced.csv')
results_out = os.path.join(current_dir, 'xgboost_final_v4_results.md')
model_out = os.path.join(current_dir, 'xgboost_final_model.json')

try:
    # 1. Veri Setini Yükle
    print(f"'{input_file_path}' yükleniyor...")
    df = pd.read_csv(input_file_path)
    X = df.drop('is_churn', axis=1)
    y = df['is_churn']

    # 2. Eğitim ve Test Ayırma
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Dengesizlik oranı
    scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
    print(f"Scale Pos Weight: {scale_pos_weight:.2f}")

    # 3. XGBoost Modeli (Önceki en iyi parametreler + Yeni Özellikler)
    # Parametreleri biraz daha güçlendiriyoruz (n_estimators artırıldı)
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight,
        learning_rate=0.1,  # Biraz daha yavaş ve dikkatli öğrensin
        max_depth=8,
        n_estimators=500,   # Daha fazla ağaç
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    print("\nModel eğitiliyor...")
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    print(f"Eğitim Tamamlandı. Süre: {end_time - start_time:.2f}s")
    
    # Modeli kaydet
    model.save_model(model_out)

    # 4. Değerlendirme ve Özellik Önemi
    print("\n--- Özellik Önemi Analizi ---")
    importance = model.feature_importances_
    feature_names = X.columns
    feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False)
    
    print("En Önemli 10 Özellik:")
    print(feat_imp_df.head(10))
    
    # 5. Threshold Tuning
    print("\n--- Eşik Değeri (Threshold) Optimizasyonu ---")
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    best_f1 = 0
    best_thr = 0.5

    for thr in thresholds:
        y_pred_thr = (y_pred_prob >= thr).astype(int)
        rep = classification_report(y_test, y_pred_thr, output_dict=True, zero_division=0)
        f1 = rep['1']['f1-score']
        print(f"Threshold: {thr} -> F1: {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    print(f"\nEn iyi Eşik Değeri: {best_thr} (F1: {best_f1:.4f})")

    # Final Rapor
    y_pred_final = (y_pred_prob >= best_thr).astype(int)
    final_rep = classification_report(y_test, y_pred_final, target_names=['Not Churn (0)', 'Churn (1)'])
    
    print("\n--- Final XGBoost (V4) Sonuçları ---")
    print(final_rep)
    
    with open(results_out, 'w') as f:
        f.write("# Final XGBoost (V4 - Advanced Features) Sonuçları\n\n")
        f.write(f"En İyi Eşik Değeri: {best_thr}\n\n")
        f.write("## En Önemli 10 Özellik\n")
        f.write(f"```\n{feat_imp_df.head(10).to_string()}\n```\n\n")
        f.write("## Sınıflandırma Raporu\n")
        f.write(f"```\n{final_rep}\n```\n")

except Exception as e:
    print(f"Bir hata oluştu: {e}")
