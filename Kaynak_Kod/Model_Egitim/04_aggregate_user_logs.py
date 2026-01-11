# 04_aggregate_user_logs.py
import pandas as pd
import os
import time

print("--- user_logs.csv dosyası özetleniyor (aggregation) ---")

# Dosya yollarını tanımla
current_dir = os.getcwd()
data_dir = os.path.join(current_dir, 'kkbox-churn-prediction-challenge')
user_logs_file_path = os.path.join(data_dir, 'user_logs.csv')
output_file_path = os.path.join(current_dir, 'user_logs_aggregated.csv')

# Parça boyutu (her seferde okunacak satır sayısı)
chunk_size = 10_000_000 # Bellek kullanımını dengede tutmak için 10 milyon satır
final_agg_df = None

start_time = time.time()

try:
    # user_logs.csv dosyasını parçalar halinde oku
    print(f"'{user_logs_file_path}' parçalar halinde yükleniyor ve işleniyor...")
    chunk_iterator = pd.read_csv(user_logs_file_path, chunksize=chunk_size)

    for i, chunk in enumerate(chunk_iterator):
        print(f"Parça {i+1} işleniyor...")
        
        # Her bir parça içinde msno'ya göre grupla ve sum (toplam) al
        chunk_agg = chunk.groupby('msno').sum()
        
        # Eğer bu ilk parçaysa, final_agg_df'i başlat
        if final_agg_df is None:
            final_agg_df = chunk_agg
        else:
            # Eğer değilse, yeni parçanın özetini final özetine ekle
            # Aynı msno'lar için değerler toplanacak
            final_agg_df = final_agg_df.add(chunk_agg, fill_value=0)

    print("\nTüm parçalar işlendi. Son DataFrame oluşturuldu.")
    
    # 'date' sütunu anlamsız olduğu için (toplamı alındı) çıkar
    if 'date' in final_agg_df.columns:
        final_agg_df = final_agg_df.drop(columns=['date'])

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
