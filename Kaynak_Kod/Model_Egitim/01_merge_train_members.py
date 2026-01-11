# 01_merge_train_members.py
import pandas as pd
import os

print("--- train.csv ve members_v3.csv dosyaları birleştiriliyor ---")

# Mevcut çalışma dizinini al
current_dir = os.getcwd()
data_dir = os.path.join(current_dir, 'kkbox-churn-prediction-challenge')

# Dosya yollarını tanımla
train_file_path = os.path.join(data_dir, 'train.csv')
members_file_path = os.path.join(data_dir, 'members_v3.csv')
output_file_path = os.path.join(current_dir, 'train_member_merged.csv') # Birleştirilmiş dosyanın kaydedileceği yer

try:
    # train.csv dosyasını yükle
    print(f"'{train_file_path}' yükleniyor...")
    train_df = pd.read_csv(train_file_path)
    print(f"train_df boyutu: {train_df.shape}")

    # members_v3.csv dosyasını yükle
    print(f"'{members_file_path}' yükleniyor...")
    members_df = pd.read_csv(members_file_path)
    print(f"members_df boyutu: {members_df.shape}")

    # msno sütunu üzerinden sol birleştirme (left merge) yap
    # Sol birleştirme, train_df'deki tüm kullanıcıları korur ve members_df'den eşleşen bilgileri ekler.
    # Eğer members_df'de eşleşen bir msno yoksa, o kullanıcının members_df'den gelen sütunları NaN (Not a Number) olacaktır.
    print("train_df ve members_df birleştiriliyor...")
    merged_df = pd.merge(train_df, members_df, on='msno', how='left')
    print(f"Birleştirilmiş DataFrame boyutu: {merged_df.shape}")

    print("\nBirleştirilmiş DataFrame'in ilk 5 satırı (merged_df.head()):\n")
    print(merged_df.head())
    print("-" * 50)

    print("\nBirleştirilmiş DataFrame hakkında genel bilgi (merged_df.info()):\n")
    merged_df.info()
    print("-" * 50)

    # Birleştirilmiş DataFrame'i yeni bir CSV dosyasına kaydet
    print(f"\nBirleştirilmiş DataFrame '{output_file_path}' olarak kaydediliyor...")
    merged_df.to_csv(output_file_path, index=False)
    print("Kaydetme tamamlandı.")

except FileNotFoundError as e:
    print(f"Hata: Dosya bulunamadı - {e.filename}. Lütfen dosya yollarını kontrol edin.")
except Exception as e:
    print(f"Bir hata oluştu: {e}")
