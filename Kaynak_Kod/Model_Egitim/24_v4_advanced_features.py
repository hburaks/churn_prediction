import pandas as pd
import numpy as np
import os
import time

print("--- İleri Seviye Özellik Mühendisliği (Adım 24-V4) ---")

current_dir = os.getcwd()
# Girdi dosyaları
model_ready_path = os.path.join(current_dir, 'model_ready_reduced_dataset_v2.csv')
transactions_path = os.path.join(current_dir, 'kkbox-churn-prediction-challenge/transactions.csv')
user_logs_v2_path = os.path.join(current_dir, 'kkbox-churn-prediction-challenge/data 4/churn_comp_refresh/user_logs_v2.csv')
output_path = os.path.join(current_dir, 'model_ready_v4_advanced.csv')

try:
    start_time = time.time()
    
    # 1. Mevcut Veri Setini Yükle
    print("Mevcut veri seti yükleniyor...")
    df_main = pd.read_csv(model_ready_path)
    # msno sütunu model_ready dosyasında olmayabilir (eğitim için çıkarılmıştı).
    # Bu yüzden train.csv ile birleştirip msno'yu geri getirmemiz lazım.
    train_path = os.path.join(current_dir, 'kkbox-churn-prediction-challenge/train.csv')
    df_train = pd.read_csv(train_path, usecols=['msno'])
    df_main['msno'] = df_train['msno'] # Sırası bozulmadıysa bu çalışır
    print(f"Ana veri boyutu: {df_main.shape}")

    # 2. Üyelik Bitiş Tarihi (Expire Date) Özelliği
    print("\nTransactions dosyasından 'membership_expire_date' işleniyor...")
    # Sadece msno ve expire_date okuyalım
    df_trans = pd.read_csv(transactions_path, usecols=['msno', 'membership_expire_date'])
    
    # Her kullanıcının EN SON (maksimum) bitiş tarihini bul
    # Tarih formatını düzelt
    df_trans['membership_expire_date'] = pd.to_datetime(df_trans['membership_expire_date'], format='%Y%m%d', errors='coerce')
    
    last_expire = df_trans.groupby('msno')['membership_expire_date'].max().reset_index()
    last_expire.columns = ['msno', 'last_expire_date']
    
    # Ana veriye birleştir
    df_main = pd.merge(df_main, last_expire, on='msno', how='left')
    
    # Gün farkını hesapla (Referans tarih: 2017-03-31 - Train setinin sonu varsayımı)
    ref_date = pd.to_datetime('2017-03-31')
    df_main['days_to_expire'] = (df_main['last_expire_date'] - ref_date).dt.days
    
    # Eksik değerleri doldur (Ortalama veya 0)
    df_main['days_to_expire'] = df_main['days_to_expire'].fillna(df_main['days_to_expire'].median())
    
    # Tarih sütununu artık kaldırabiliriz
    df_main = df_main.drop('last_expire_date', axis=1)
    print("- 'days_to_expire' özelliği eklendi.")

    # 3. Trend Özellikleri (User Logs V2'den)
    print("\nUser Logs V2 üzerinden Trend (Değişim) özellikleri hesaplanıyor...")
    df_logs = pd.read_csv(user_logs_v2_path)
    df_logs['date'] = pd.to_datetime(df_logs['date'], format='%Y%m%d', errors='coerce')
    
    # Referans tarih (V2 loglarının sonu)
    max_log_date = df_logs['date'].max()
    print(f"En son log tarihi: {max_log_date}")
    
    # İki periyoda böl: Son 14 gün vs Önceki 14 gün
    cutoff_1 = max_log_date - pd.Timedelta(days=14)
    cutoff_2 = max_log_date - pd.Timedelta(days=28)
    
    # Son 14 gün (Recent)
    mask_recent = (df_logs['date'] > cutoff_1)
    # Önceki 14 gün (Previous)
    mask_previous = (df_logs['date'] <= cutoff_1) & (df_logs['date'] > cutoff_2)
    
    # Özetle (Toplam saniye üzerinden)
    cols_to_sum = ['total_secs', 'num_unq', 'num_100']
    
    # Recent Stats
    recent_stats = df_logs[mask_recent].groupby('msno')[cols_to_sum].sum().add_suffix('_recent')
    # Previous Stats
    prev_stats = df_logs[mask_previous].groupby('msno')[cols_to_sum].sum().add_suffix('_prev')
    
    # Birleştir
    trend_stats = pd.merge(recent_stats, prev_stats, on='msno', how='outer').fillna(0)
    
    # Oranları Hesapla (Trend = Recent / Previous)
    # 0'a bölme hatasını önlemek için +1 ekliyoruz (Laplace Smoothing benzeri)
    for col in cols_to_sum:
        trend_stats[f'{col}_trend'] = (trend_stats[f'{col}_recent'] + 1) / (trend_stats[f'{col}_prev'] + 1)
    
    # Sadece trend sütunlarını al
    trend_cols = [c for c in trend_stats.columns if '_trend' in c]
    df_trends = trend_stats[trend_cols].reset_index()
    
    # Ana veriye birleştir
    df_main = pd.merge(df_main, df_trends, on='msno', how='left')
    
    # Log kaydı olmayanlar için trend 1 (değişim yok) varsayılabilir
    for col in trend_cols:
        df_main[col] = df_main[col].fillna(1.0)
        
    print(f"- {len(trend_cols)} adet trend özelliği eklendi: {trend_cols}")

    # 4. Temizlik ve Kayıt
    # msno'yu tekrar çıkar (eğitim için)
    df_final = df_main.drop('msno', axis=1)
    
    print(f"\nFinal Veri Seti Boyutu: {df_final.shape}")
    df_final.to_csv(output_path, index=False)
    print(f"Veri seti kaydedildi: {output_path}")
    print(f"Toplam Süre: {time.time() - start_time:.2f}s")

except Exception as e:
    print(f"Bir hata oluştu: {e}")
