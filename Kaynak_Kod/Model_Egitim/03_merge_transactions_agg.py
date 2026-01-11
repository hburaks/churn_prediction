# 03_merge_transactions_agg.py
import pandas as pd
import os

print("--- train_member_merged.csv ve transactions_aggregated.csv dosyaları birleştiriliyor ---")

# Mevcut çalışma dizinini al
current_dir = os.getcwd()

# Dosya yollarını tanımla
train_member_merged_path = os.path.join(current_dir, 'train_member_merged.csv')
transactions_agg_path = os.path.join(current_dir, 'transactions_aggregated.csv')
output_file_path = os.path.join(current_dir, 'train_members_transactions_merged.csv')

try:
    # train_member_merged.csv dosyasını yükle
    print(f"'{train_member_merged_path}' yükleniyor...")
    main_df = pd.read_csv(train_member_merged_path)
    print(f"main_df boyutu: {main_df.shape}")

    # transactions_aggregated.csv dosyasını yükle
    print(f"'{transactions_agg_path}' yükleniyor...")
    transactions_agg_df = pd.read_csv(transactions_agg_path)
    print(f"transactions_agg_df boyutu: {transactions_agg_df.shape}")

    # msno sütunu üzerinden sol birleştirme (left merge) yap
    print("Ana DataFrame ve özetlenmiş işlemler birleştiriliyor...")
    merged_df = pd.merge(main_df, transactions_agg_df, on='msno', how='left')
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
