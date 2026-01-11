# 11_aggregate_user_logs_v2.py
import pandas as pd
import os
import numpy as np
import time

print("--- user_logs.csv dosyası kapsamlı olarak özetleniyor (aggregation v2) ---")

# Dosya yollarını tanımla
current_dir = os.getcwd()
data_dir = os.path.join(current_dir, 'kkbox-churn-prediction-challenge')
user_logs_file_path = os.path.join(data_dir, 'user_logs.csv')
output_file_path = os.path.join(current_dir, 'user_logs_aggregated_v2.csv')

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

try:
    # user_logs.csv dosyasını parçalar halinde oku
    print(f"'{user_logs_file_path}' parçalar halinde yükleniyor ve işleniyor...")
    chunk_iterator = pd.read_csv(user_logs_file_path, chunksize=chunk_size)

    for i, chunk in enumerate(chunk_iterator):
        print(f"Parça {i+1} işleniyor...")
        
        # total_secs sütunundaki negatif değerleri 0 yap (temizlik)
        chunk.loc[chunk['total_secs'] < 0, 'total_secs'] = 0

        # Her bir parça içinde msno'ya göre grupla ve istatistikleri hesapla
        chunk_agg = chunk.groupby('msno')[log_cols].agg(['sum', 'count', 'min', 'max'])
        
        # Çok seviyeli sütun isimlerini düzleştir
        chunk_agg.columns = ['_'.join(col).strip() for col in chunk_agg.columns.values]
        
        # Her kullanıcı için biriktir
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

    print("\nTüm parçalar işlendi. Son DataFrame oluşturuluyor.")

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
    final_agg_df = final_agg_df.set_index('msno') # msno'yu index yap

    print("\nÖzetlenmiş DataFrame'in ilk 5 satırı (final_agg_df.head()):\n")
    print(final_agg_df.head())
    print("-" * 50)

    print("\nÖzetlenmiş DataFrame hakkında genel bilgi (final_agg_df.info()):\n")
    final_agg_df.info()
    print("-" * 50)

    # Özetlenmiş DataFrame'i yeni bir CSV dosyasına kaydet
    print(f"\nÖzetlenmiş DataFrame '{output_file_path}' olarak kaydediliyor...")
    final_agg_df.to_csv(output_file_path)
    print("Kaydetme tamamlandı.")

    end_time = time.time()
    print(f"\nToplam İşlem Süresi: {end_time - start_time:.2f} saniye")

except FileNotFoundError as e:
    print(f"Hata: Dosya bulunamadı - {e.filename}. Lütfen dosya yollarını kontrol edin.")
except Exception as e:
    print(f"Bir hata oluştu: {e}")
