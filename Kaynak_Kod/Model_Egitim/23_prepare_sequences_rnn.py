import pandas as pd
import numpy as np
import os
import time

print("--- RNN/LSTM İçin Zaman Serisi Veri Hazırlığı (V2 Verisiyle) ---")

# Dosya yolları
current_dir = os.getcwd()
user_logs_v2_path = os.path.join(current_dir, 'kkbox-churn-prediction-challenge/data 4/churn_comp_refresh/user_logs_v2.csv')
train_path = os.path.join(current_dir, 'kkbox-churn-prediction-challenge/train.csv')
X_rnn_out = os.path.join(current_dir, 'rnn_X_sequences.npz')
y_rnn_out = os.path.join(current_dir, 'rnn_y_targets.npy')

# Ayarlar
SEQUENCE_LENGTH = 14
FEATURES = ['num_25', 'num_50', 'num_75', 'num_985', 'num_100', 'num_unq', 'total_secs']

try:
    # 1. Kullanıcı Listesini Al
    print("Eğitim seti kullanıcıları alınıyor...")
    train_df = pd.read_csv(train_path)
    train_msnos = set(train_df['msno'])
    y_mapping = dict(zip(train_df['msno'], train_df['is_churn']))
    print(f"- {len(train_msnos)} kullanıcı hedef listesinde.")

    # 2. Logları Oku (V2 dosyası daha küçük olduğu için daha hızlı)
    print(f"\n'{user_logs_v2_path}' yükleniyor...")
    start_time = time.time()
    
    # user_logs_v2.csv'yi yükle (Yaklaşık 1.3 GB, belleğe sığabilir)
    logs_v2_df = pd.read_csv(user_logs_v2_path)
    print(f"Log satır sayısı: {len(logs_v2_df)}")

    # Sadece eğitim setindeki kullanıcıları filtrele
    logs_v2_df = logs_v2_df[logs_v2_df['msno'].isin(train_msnos)]
    print(f"Filtrelenmiş log satır sayısı: {len(logs_v2_df)}")

    # Tarihe göre sırala (Önemli!)
    print("Tarihe göre sıralanıyor...")
    logs_v2_df = logs_v2_df.sort_values(by=['msno', 'date'])

    # 3. Sequence Oluşturma
    print("\nSekanslar oluşturuluyor...")
    X_list = []
    y_list = []
    
    # msno'ya göre grupla ve her grup için son N günü al
    for msno, group in logs_v2_df.groupby('msno'):
        seq = group[FEATURES].values
        
        # Son SEQUENCE_LENGTH günü al
        if len(seq) > SEQUENCE_LENGTH:
            seq = seq[-SEQUENCE_LENGTH:]
        
        # Padding
        if len(seq) < SEQUENCE_LENGTH:
            padding = np.zeros((SEQUENCE_LENGTH - len(seq), len(FEATURES)))
            seq = np.vstack([padding, seq])
        
        X_list.append(seq)
        y_list.append(y_mapping[msno])

    X_rnn = np.array(X_list, dtype='float32')
    y_rnn = np.array(y_list, dtype='int8')

    # 4. Kaydet
    print(f"\nSonuçlar kaydediliyor... X: {X_rnn.shape}")
    np.savez_compressed(X_rnn_out, data=X_rnn)
    np.save(y_rnn_out, y_rnn)

    print(f"\n--- Adım 23 Tamamlandı ---")
    print(f"Süre: {time.time() - start_time:.2f}s")

except Exception as e:
    print(f"Bir hata oluştu: {e}")