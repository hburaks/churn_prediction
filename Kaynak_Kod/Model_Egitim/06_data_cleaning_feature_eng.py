# 06_data_cleaning_feature_eng.py
import pandas as pd
import os
import numpy as np

print("--- Veri Temizliği ve Özellik Mühendisliği Başlatılıyor ---")

# Dosya yollarını tanımla
current_dir = os.getcwd()
input_file_path = os.path.join(current_dir, 'final_train_dataset.csv')
output_file_path = os.path.join(current_dir, 'model_ready_dataset.csv')

try:
    # Nihai birleştirilmiş veri setini yükle
    print(f"'{input_file_path}' yükleniyor...")
    df = pd.read_csv(input_file_path)
    print(f"Orijinal DataFrame boyutu: {df.shape}")

    # 1. Eksik Değerleri Doldurma ve Veri Kalitesi Sorunlarını Düzeltme
    print("\n1. Eksik değerler dolduruluyor ve veri kalitesi sorunları düzeltiliyor...")
    
    # 'bd' (yaş) sütununu temizle
    # Mantıksız yaş değerlerini (10'dan küçük veya 100'den büyük) NaN olarak ayarla
    df.loc[(df['bd'] < 10) | (df['bd'] > 100), 'bd'] = np.nan
    # Kalan NaN değerleri, yaş sütununun medyanı ile doldur
    median_age = df['bd'].median()
    df['bd'] = df['bd'].fillna(median_age)
    print(f"- 'bd' (yaş) sütunu temizlendi. Anlamsız yaşlar medyan ({median_age}) ile dolduruldu.")

    # 'gender' sütunundaki NaN değerleri 'unknown' olarak doldur
    df['gender'] = df['gender'].fillna('unknown')
    print("- 'gender' sütunundaki eksik değerler 'unknown' olarak dolduruldu.")
    
    # 'city' ve 'registered_via' sütunlarındaki NaN değerleri en sık tekrar eden değer (mode) ile doldur
    mode_city = df['city'].mode()[0]
    df['city'] = df['city'].fillna(mode_city)
    print(f"- 'city' sütunundaki eksik değerler en sık tekrar eden şehir ({mode_city}) ile dolduruldu.")
    
    mode_reg = df['registered_via'].mode()[0]
    df['registered_via'] = df['registered_via'].fillna(mode_reg)
    print(f"- 'registered_via' sütunundaki eksik değerler en sık tekrar eden yöntem ({mode_reg}) ile dolduruldu.")

    # 3. Yeni Özellikler Türetme (Feature Engineering)
    print("\n3. Yeni özellikler türetiliyor...")
    # 'registration_init_time' sütununu datetime formatına çevir
    df['registration_init_time'] = pd.to_datetime(df['registration_init_time'], format='%Y%m%d', errors='coerce')
    # Veri setindeki en son tarihi referans alarak üyelik süresini gün olarak hesapla
    latest_date = pd.to_datetime('20170301', format='%Y%m%d')
    df['membership_days'] = (latest_date - df['registration_init_time']).dt.days
    # Tarih sütunundaki olası NaN değerleri (coerce sonrası) medyan ile doldur
    df['membership_days'] = df['membership_days'].fillna(df['membership_days'].median())
    print("- 'membership_days' (üyelik süresi) özelliği oluşturuldu.")

    # 4. Kategorik Sütunları Sayısala Çevirme (One-Hot Encoding)
    print("\n4. Kategorik sütunlar sayısala çevriliyor...")
    categorical_cols = ['city', 'gender', 'registered_via']
    df = pd.get_dummies(df, columns=categorical_cols, dummy_na=False)
    print("- 'city', 'gender', 'registered_via' sütunları one-hot encoding ile dönüştürüldü.")

    # 5. Gereksiz Sütunları Kaldırma
    print("\n5. Gereksiz sütunlar kaldırılıyor...")
    df = df.drop(columns=['msno', 'registration_init_time'])
    print("- 'msno' ve 'registration_init_time' sütunları kaldırıldı.")

    print("\n--- Veri Temizliği ve Özellik Mühendisliği Tamamlandı ---")
    print(f"Son DataFrame boyutu: {df.shape}")
    print("\nSon DataFrame hakkında genel bilgi (df.info()):\n")
    df.info()

    # Modellenmeye hazır DataFrame'i yeni bir CSV dosyasına kaydet
    print(f"\nModellenmeye hazır DataFrame '{output_file_path}' olarak kaydediliyor...")
    df.to_csv(output_file_path, index=False)
    print("Kaydetme tamamlandı.")

except FileNotFoundError as e:
    print(f"Hata: Dosya bulunamadı - {e.filename}. Lütfen dosya yollarını kontrol edin.")
except Exception as e:
    print(f"Bir hata oluştu: {e}")
