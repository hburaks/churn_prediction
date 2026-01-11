import pandas as pd
import numpy as np
import os
import random

print("--- Demo Verisi Zenginleştirme (Sadık Kullanıcı Ekleme) ---")

current_dir = os.getcwd()
trans_path = os.path.join(current_dir, 'kkbox-churn-prediction-challenge/transactions.csv')
demo_db_path = os.path.join(current_dir, 'model_ready_lite_sample.csv')
output_path = os.path.join(current_dir, 'model_ready_lite_sample_enriched.csv')

try:
    # 1. Mevcut Veritabanını Yükle
    print("Mevcut demo verisi yükleniyor...")
    df_current = pd.read_csv(demo_db_path)
    print(f"Mevcut kayıt sayısı: {len(df_current)}")
    
    # Kolon yapısını al
    columns = df_current.columns.tolist()
    
    # 2. Transactions Dosyasından Veri Bulunamadığı İçin Sentetik Veri Üret
    print("Sentetik 'Sadık Kullanıcı' verisi üretiliyor...")
    
    new_rows = []
    num_synthetic = 100
    
    for i in range(num_synthetic):
        new_user = {}
        
        # Temel özellikler
        for col in columns:
            new_user[col] = 0
            
        # Sadık Profil Özellikleri
        # Days to Expire: 30 gün ile 180 gün arası (Gelecek)
        new_user['days_to_expire'] = random.randint(30, 180)
        
        # Hedef: Churn Değil
        new_user['is_churn'] = 0
        
        # Davranışlar
        new_user['is_cancel_sum'] = 0 # Hiç iptal etmemiş
        new_user['is_auto_renew_max'] = 1 # Otomatik ödeme açık
        new_user['total_transactions'] = random.randint(12, 60) # Düzenli ödüyor
        new_user['membership_days'] = random.randint(365, 1500) # Eski üye
        new_user['plan_list_price_mean'] = 149 # Standart fiyat
        
        # Trendler (Stabil veya Artan)
        new_user['num_100_trend'] = random.uniform(0.95, 1.1)
        new_user['total_secs_trend'] = random.uniform(0.95, 1.1)
        new_user['num_unq_trend'] = random.uniform(0.95, 1.1)
        
        new_rows.append(new_user)
        
    df_new = pd.DataFrame(new_rows)
    print(f"Üretilen sentetik kullanıcı sayısı: {len(df_new)}")
    
    # 4. Birleştir (Yeni kullanıcıları EN BAŞA koy)
    df_final = pd.concat([df_new, df_current], ignore_index=True)
    
    # 5. Kaydet
    # Orijinal dosyanın üzerine yazmak yerine yedeğini alıp yazalım mı?
    # Direkt backend'in kullandığı isimle kaydedelim ki config değiştirmeyelim.
    df_final.to_csv(demo_db_path, index=False)
    
    print(f"\nİşlem Tamamlandı.")
    print(f"Yeni veritabanı boyutu: {len(df_final)}")
    print(f"İlk 100 satır (ID 0-99) artık 'Güvenli/Aktif' kullanıcılar.")

except Exception as e:
    print(f"Hata: {e}")
