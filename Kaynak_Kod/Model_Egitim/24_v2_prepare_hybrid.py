import pandas as pd
import numpy as np
import os
import time

print("--- Hibrit Model İçin Veri Hizalama (Adım 24-V2) ---")

current_dir = os.getcwd()
tabular_data_path = os.path.join(current_dir, 'model_ready_reduced_dataset_v2.csv')
train_path = os.path.join(current_dir, 'kkbox-churn-prediction-challenge/train.csv')
rnn_X_path = os.path.join(current_dir, 'rnn_X_sequences.npz')

try:
    # 1. Tablolu veriyi yükle ve msno ile eşleştir
    print("Tablolu veriler ve kullanıcı listesi yükleniyor...")
    df_tabular = pd.read_csv(tabular_data_path)
    df_train = pd.read_csv(train_path)
    
    # msno'yu geri ekle (hizalama için şart)
    df_tabular['msno'] = df_train['msno']
    
    # 2. Sekans verisindeki kullanıcıları belirle
    # Adım 23'te rnn verisi msno'ya göre sıralı (alphabetical) oluşturulmuştu
    user_logs_v2_path = os.path.join(current_dir, 'kkbox-churn-prediction-challenge/data 4/churn_comp_refresh/user_logs_v2.csv')
    logs_v2_msnos = sorted(pd.read_csv(user_logs_v2_path, usecols=['msno'])['msno'].unique())
    logs_v2_msnos = [m for m in logs_v2_msnos if m in set(df_train['msno'])]
    print(f"Sekans verisindeki benzersiz kullanıcı sayısı: {len(logs_v2_msnos)}")

    # 3. Tablolu veriyi sekans verisindeki kullanıcılar için filtrele ve sırala
    print("Tablolu veri sekans verisiyle hizalanıyor...")
    df_tabular_filtered = df_tabular[df_tabular['msno'].isin(logs_v2_msnos)].sort_values('msno')
    
    # msno ve is_churn sütunlarını ayır (sadece öznitelikler kalsın)
    y_hybrid = df_tabular_filtered['is_churn'].values
    X_tabular_hybrid = df_tabular_filtered.drop(['msno', 'is_churn'], axis=1).values
    
    # 4. Sekans verisini yükle (Zaten msno'ya göre sıralı kaydedildi)
    print("Sekans verisi yükleniyor...")
    with np.load(rnn_X_path) as data:
        X_rnn_hybrid = data['data']

    print(f"\nHizalama Tamamlandı:")
    print(f"Tablolu Veri Boyutu: {X_tabular_hybrid.shape}")
    print(f"Sekans Veri Boyutu:  {X_rnn_hybrid.shape}")
    print(f"Hedef Değişken:      {y_hybrid.shape}")

    # 5. Kaydet
    print("\nHibrit eğitim verileri kaydediliyor...")
    np.save('hybrid_X_tabular.npy', X_tabular_hybrid.astype('float32'))
    np.save('hybrid_X_rnn.npy', X_rnn_hybrid.astype('float32'))
    np.save('hybrid_y.npy', y_hybrid.astype('int8'))

    print("--- Hazırlık Tamamlandı ---")

except Exception as e:
    print(f"Bir hata oluştu: {e}")
