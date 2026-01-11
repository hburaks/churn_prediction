import xgboost as xgb
import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

print("--- Final Model Temizliği ve Sadeleştirme (Adım 25) ---")

current_dir = os.getcwd()
data_path = os.path.join(current_dir, 'model_ready_v4_advanced.csv')
model_path = os.path.join(current_dir, 'xgboost_final_model.json')
output_model_path = os.path.join(current_dir, 'xgboost_final_model_lite.json')
feature_list_path = os.path.join(current_dir, 'feature_list.json')

try:
    # 1. Veriyi Yükle
    print(f"Veri yükleniyor: {data_path}")
    df = pd.read_csv(data_path)
    X = df.drop('is_churn', axis=1)
    y = df['is_churn']
    
    feature_names = list(X.columns)
    print(f"Başlangıç özellik sayısı: {len(feature_names)}")

    # 2. Mevcut Modeli Yükle ve Özellik Önemini Al
    print("Mevcut model yükleniyor...")
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    
    # Özellik önemlerini al (gain veya weight)
    importance_type = 'gain' # 'weight', 'gain', 'cover'
    scores = model.get_booster().get_score(importance_type=importance_type)
    
    # Feature map olmadığı için skor anahtarları f0, f1... olabilir.
    # XGBoost'un feature names eşleştirmesini kullanalım
    # Ancak sklearn API ile load_model yapınca feature_names_in_ korunmuş olmalı
    
    try:
        current_features = model.feature_names_in_
        importances = model.feature_importances_
        feature_imp_dict = dict(zip(current_features, importances))
    except:
        print("Uyarı: Feature isimleri modelden alınamadı, manuel eşleştiriliyor...")
        # Alternatif: f0 -> feature_names[0]
        feature_imp_dict = {}
        for k, v in scores.items():
            idx = int(k[1:]) # f12 -> 12
            if idx < len(feature_names):
                feature_imp_dict[feature_names[idx]] = v

    # Sıralı liste yap
    sorted_features = sorted(feature_imp_dict.items(), key=lambda item: item[1], reverse=True)
    
    print("\n--- Özellik Önem Sıralaması (İlk 10) ---")
    for f, s in sorted_features[:10]:
        print(f"{f}: {s:.4f}")

    # 3. Eleme (Threshold Belirleme)
    # Toplam önemin %99'unu oluşturan özellikleri tutalım veya
    # Mutlak eşik: 0.005'ten küçük olanları atalım (Genelde 0 ile 1 arasıdır importances_)
    
    threshold = 0.005 # %0.5'ten az etkisi olanları at
    selected_features = [f for f, s in sorted_features if s >= threshold]
    
    # "is_cancel_sum" gibi kritiklerin yanlışlıkla elenmemesini garantiye alalım (gerçi yüksek çıkacaktır)
    # Ayrıca, eğer selected_features çok az kalırsa (örn < 5), en iyi 10'u alalım.
    if len(selected_features) < 5:
        selected_features = [f for f, s in sorted_features[:10]]
        
    print(f"\nSeçilen Özellik Sayısı: {len(selected_features)}")
    print(f"Elenen Özellik Sayısı: {len(feature_names) - len(selected_features)}")
    print(f"Kalan Özellikler: {selected_features}")
    
    # Listeyi kaydet (Backend için)
    with open(feature_list_path, 'w') as f:
        json.dump(selected_features, f)
    print(f"Seçilen özellik listesi kaydedildi: {feature_list_path}")

    # 4. Veriyi Sadeleştir ve Yeniden Eğitim
    print("\nModel seçilen özelliklerle yeniden eğitiliyor...")
    X_reduced = X[selected_features]
    
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale Pos Weight (Sınıf dengesizliği için)
    ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
    
    # Yeni "Hafif" Model
    model_lite = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=200, # V4'teki ayarlar
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=ratio,
        random_state=42,
        n_jobs=-1
    )
    
    model_lite.fit(X_train, y_train, verbose=False)
    
    # 5. Performans Kontrolü
    y_pred_prob = model_lite.predict_proba(X_test)[:, 1]
    # En iyi threshold V4'te 0.9 çıkmıştı, burada da kontrol edelim
    best_thresh = 0.9 
    y_pred = (y_pred_prob >= best_thresh).astype(int)
    
    print("\n--- LITE Model Performansı (Eşik: 0.9) ---")
    print(classification_report(y_test, y_pred))
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
    
    # 6. Kayıt
    model_lite.save_model(output_model_path)
    print(f"Hafif model kaydedildi: {output_model_path}")
    
    # Sadeleştirilmiş veriyi kaydetmeye gerek var mı? 
    # Backend zaten feature_list.json ile ham veriden (veya full csv'den) gerekli sütunları çekecek.
    # Ama analiz kolaylığı için test setinin bir kısmını kaydedelim.
    sample_data = X_reduced.copy()
    sample_data['is_churn'] = y # Hedefi geri koy
    sample_data.to_csv('model_ready_lite_sample.csv', index=False)
    print("Örnek sadeleştirilmiş veri seti (model_ready_lite_sample.csv) kaydedildi.")

except Exception as e:
    print(f"Hata: {e}")
    import traceback
    traceback.print_exc()
