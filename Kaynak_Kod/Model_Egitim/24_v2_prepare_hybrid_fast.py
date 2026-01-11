import pandas as pd
import numpy as np
import os
import time

print("--- Hibrit Model İçin Hızlı Veri Hizalama (Adım 24-V2) ---")

current_dir = os.getcwd()
tabular_data_path = os.path.join(current_dir, 'model_ready_reduced_dataset_v2.csv')
train_path = os.path.join(current_dir, 'kkbox-churn-prediction-challenge/train.csv')
rnn_X_path = os.path.join(current_dir, 'rnn_X_sequences.npz')

try:
    start_time = time.time()
    
    # 1. Sadece msno'ları içeren listeleri yükle (Tüm dosyayı okumaktan kaçın)
    print("Kullanıcı listeleri yükleniyor...")
    df_train = pd.read_csv(train_path, usecols=['msno'])
    
    # user_logs_v2'deki msno'ları Adım 23'teki mantıkla belirle
    user_logs_v2_path = os.path.join(current_dir, 'kkbox-churn-prediction-challenge/data 4/churn_comp_refresh/user_logs_v2.csv')
    logs_v2_msnos = pd.read_csv(user_logs_v2_path, usecols=['msno'])['msno'].unique()
    
    # train.csv'de de olan kullanıcıları bul ve alfabetik sırala (Adım 23'teki groupby sırası)
    train_msnos_set = set(df_train['msno'])
    common_msnos = sorted([m for m in logs_v2_msnos if m in train_msnos_set])
    common_msnos_set = set(common_msnos)
    print(f"Eşleşen kullanıcı sayısı: {len(common_msnos)}")

    # 2. Tablolu veriyi yükle
    print("Tablolu veriler yükleniyor...")
    df_tabular = pd.read_csv(tabular_data_path)
    df_tabular['msno'] = df_train['msno']
    
    # Filtrele ve sırala
    print("Veriler hizalanıyor...")
    df_tabular_filtered = df_tabular[df_tabular['msno'].isin(common_msnos_set)].sort_values('msno')
    
    X_tabular_hybrid = df_tabular_filtered.drop(['msno', 'is_churn'], axis=1).values
    y_hybrid = df_tabular_filtered['is_churn'].values

    # 3. Sekans verisini yükle
    print("Sekans verileri (npz) yükleniyor...")
    with np.load(rnn_X_path) as data:
        X_rnn_hybrid = data['data']

    # Güvenlik kontrolü: Boyutlar tutuyor mu?
    if X_tabular_hybrid.shape[0] != X_rnn_hybrid.shape[0]:
        print(f"HATA: Boyut uyuşmazlığı! Tabular: {X_tabular_hybrid.shape[0]}, RNN: {X_rnn_hybrid.shape[0]}")
        # Eğer uyuşmazlık varsa küçük olan boyuta göre kırp (genelde rnn tamdır)
        min_len = min(X_tabular_hybrid.shape[0], X_rnn_hybrid.shape[0])
        X_tabular_hybrid = X_tabular_hybrid[:min_len]
        X_rnn_hybrid = X_rnn_hybrid[:min_len]
        y_hybrid = y_hybrid[:min_len]

    # 4. Kaydet
    print(f"Son boyutlar: Tabular={X_tabular_hybrid.shape}, RNN={X_rnn_hybrid.shape}")
    np.save('hybrid_X_tabular.npy', X_tabular_hybrid.astype('float32'))
    np.save('hybrid_X_rnn.npy', X_rnn_hybrid.astype('float32'))
    np.save('hybrid_y.npy', y_hybrid.astype('int8'))

    print(f"--- Hazırlık Tamamlandı ({time.time() - start_time:.2f}s) ---")

except Exception as e:
    print(f"Bir hata oluştu: {e}")
