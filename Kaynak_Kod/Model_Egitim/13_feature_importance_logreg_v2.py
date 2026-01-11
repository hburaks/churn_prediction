# 13_feature_importance_logreg_v2.py
import pandas as pd
import os
import pickle

print("--- Yeni Modelin Özellik Önemini Analiz Etme (v2) ---")

# Dosya yollarını tanımla
current_dir = os.getcwd()
model_input_path = os.path.join(current_dir, 'logreg_model_v2.pkl')
data_input_path = os.path.join(current_dir, 'model_ready_dataset.csv')

try:
    # Eğitilmiş modeli yükle
    print(f"'{model_input_path}' yükleniyor...")
    with open(model_input_path, 'rb') as f:
        model = pickle.load(f)
    print("- Model yüklendi.")

    # Özellik isimlerini almak için veri setini yükle (sadece sütunlar)
    print(f"Özellik isimleri için '{data_input_path}' okunuyor...")
    df = pd.read_csv(data_input_path, nrows=0) # Sadece başlıkları oku
    feature_names = df.drop('is_churn', axis=1).columns
    print(f"- {len(feature_names)} adet özellik bulundu.")

    # Model katsayılarını al
    coefficients = model.coef_[0]

    # Özellikler ve katsayıları ile bir DataFrame oluştur
    feature_importance = pd.DataFrame({'feature': feature_names, 'importance': coefficients})
    
    # Mutlak değere göre sırala
    feature_importance['abs_importance'] = feature_importance['importance'].abs()
    feature_importance = feature_importance.sort_values(by='abs_importance', ascending=False)
    
    print("\n--- En Önemli 15 Özellik (v2) ---")
    print(feature_importance.head(15).to_string(index=False))

    print("\n--- En Etkisiz 15 Özellik (v2) ---")
    print(feature_importance.tail(15).to_string(index=False))

except FileNotFoundError as e:
    print(f"Hata: Dosya bulunamadı - {e.filename}")
except Exception as e:
    print(f"Bir hata oluştu: {e}")
