# 11_aggregate_user_logs_filtered.py
import pandas as pd
import os
import numpy as np
import time

print("--- user_logs.csv dosyası filtrelenerek ve kapsamlı olarak özetleniyor ---")

# Dosya yollarını tanımla
current_dir = os.getcwd()
data_dir = os.path.join(current_dir, 'kkbox-churn-prediction-challenge')
user_logs_file_path = os.path.join(data_dir, 'user_logs.csv')
train_file_path = os.path.join(data_dir, 'train.csv')
output_file_path = os.path.join(current_dir, 'user_logs_aggregated_v2.csv')

try:
    # 1. Sadece eğitim setindeki kullanıcıları al
    print("Eğitim setindeki kullanıcı kimlikleri (msno) alınıyor...")
    train_df = pd.read_csv(train_file_path)
    train_msnos = set(train_df['msno'])
    print(f"- {len(train_msnos)} adet benzersiz kullanıcı bulundu.")

    # Parça boyutu
    chunk_size = 10_000_000

    # Agregasyon için sütunlar
    log_cols = ['num_25', 'num_50', 'num_75', 'num_985', 'num_100', 'num_unq', 'total_secs']

    # Her kullanıcı için biriktirilecek sözlükler
    user_sums = {}
    user_counts = {}
    user_min_vals = {}
    user_max_vals = {}

    start_time = time.time()

    # 2. user_logs.csv dosyasını parçalar halinde oku ve işle
    print(f"\n'{user_logs_file_path}' parçalar halinde yükleniyor ve işleniyor...")
    chunk_iterator = pd.read_csv(user_logs_file_path, chunksize=chunk_size)

    for i, chunk in enumerate(chunk_iterator):
        # 3. Her parçayı, sadece eğitim setindeki kullanıcıları içerecek şekilde filtrele
        chunk_filtered = chunk[chunk['msno'].isin(train_msnos)]
        
        if not chunk_filtered.empty:
            print(f"Parça {i+1} işleniyor... (Filtrelenmiş satır sayısı: {len(chunk_filtered)})")
            
            # total_secs sütunundaki negatif değerleri 0 yap (temizlik)
            chunk_filtered.loc[chunk_filtered['total_secs'] < 0, 'total_secs'] = 0

            # 4. Filtrelenmiş parça içinde msno'ya göre grupla ve istatistikleri hesapla
            chunk_agg = chunk_filtered.groupby('msno')[log_cols].agg(['sum', 'count', 'min', 'max'])
            
            # Çok seviyeli sütun isimlerini düzleştir
            chunk_agg.columns = ['_'.join(col).strip() for col in chunk_agg.columns.values]
            
            # 5. Her kullanıcı için istatistikleri biriktir
            for msno, row in chunk_agg.iterrows():
                if msno not in user_sums:
                    user_sums[msno] = {col: 0 for col in log_cols}
                    user_counts[msno] = {col: 0 for col in log_cols}
                    user_min_vals[msno] = {col: np.inf for col in log_cols}
                    user_max_vals[msno] = {col: -np.inf for col in log_cols}

                for col in log_cols:
                    user_sums[msno][col] += row[f'{col}_sum']
                    user_counts[msno][col] += row[f'{col}_count']
                    user_min_vals[msno][col] = min(user_min_vals[msno][col], row[f'{col}_min'])
                    user_max_vals[msno][col] = max(user_max_vals[msno][col], row[f'{col}_max'])
        else:
            print(f"Parça {i+1} işleniyor... (İlgili kullanıcı bulunamadı)")

    print("\n6. Tüm parçalar işlendi. Son DataFrame oluşturuluyor.")

    # Nihai DataFrame'i oluştur
    final_data = []
    for msno in user_sums:
        row_data = {'msno': msno}
        for col in log_cols:
            total_sum = user_sums[msno][col]
            total_count = user_counts[msno][col]
            
            row_data[f'{col}_sum'] = total_sum
            row_data[f'{col}_count'] = total_count # Aktif gün sayısı olarak kullanılabilir
            row_data[f'{col}_mean'] = total_sum / total_count if total_count > 0 else 0
            row_data[f'{col}_min'] = user_min_vals[msno][col] if user_min_vals[msno][col] != np.inf else 0
            row_data[f'{col}_max'] = user_max_vals[msno][col] if user_max_vals[msno][col] != -np.inf else 0
            
        final_data.append(row_data)

    final_agg_df = pd.DataFrame(final_data)
    final_agg_df = final_agg_df.set_index('msno')

    print("\nÖzetlenmiş DataFrame'in ilk 5 satırı:\n")
    print(final_agg_df.head())
    print("-" * 50)

    print("\nÖzetlenmiş DataFrame hakkında genel bilgi:\n")
    final_agg_df.info()
    print("-" * 50)

    # 7. Özetlenmiş DataFrame'i yeni bir CSV dosyasına kaydet
    print(f"Özetlenmiş DataFrame '{output_file_path}' olarak kaydediliyor...")
    final_agg_df.to_csv(output_file_path)
    print("Kaydetme tamamlandı.")

    end_time = time.time()
    print(f"\nToplam İşlem Süresi: {end_time - start_time:.2f} saniye")

except FileNotFoundError as e:
    print(f"Hata: Dosya bulunamadı - {e.filename}. Lütfen dosya yollarını kontrol edin.")
except Exception as e:
    print(f"Bir hata oluştu: {e}")
