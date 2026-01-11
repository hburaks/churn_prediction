# 05_merge_user_logs_agg.py
import pandas as pd
import os

print("--- train_members_transactions_merged.csv ve user_logs_aggregated.csv dosyaları birleştiriliyor ---")

# Mevcut çalışma dizinini al
current_dir = os.getcwd()

# Dosya yollarını tanımla
main_merged_path = os.path.join(current_dir, 'train_members_transactions_merged.csv')
user_logs_agg_path = os.path.join(current_dir, 'user_logs_aggregated.csv')
output_file_path = os.path.join(current_dir, 'final_train_dataset.csv')

try:
    # train_members_transactions_merged.csv dosyasını yükle
    print(f"'{main_merged_path}' yükleniyor...")
    main_df = pd.read_csv(main_merged_path)
    print(f"main_df boyutu: {main_df.shape}")

    # user_logs_aggregated.csv dosyasını yükle
    print(f"'{user_logs_agg_path}' yükleniyor...")
    user_logs_agg_df = pd.read_csv(user_logs_agg_path)
    print(f"user_logs_agg_df boyutu: {user_logs_agg_df.shape}")

    # msno sütunu üzerinden sol birleştirme (left merge) yap
    print("Ana DataFrame ve özetlenmiş kullanıcı logları birleştiriliyor...")
    final_df = pd.merge(main_df, user_logs_agg_df, on='msno', how='left')
    print(f"Birleştirilmiş DataFrame boyutu: {final_df.shape}")

    print("\nBirleştirilmiş DataFrame'in ilk 5 satırı (final_df.head()):\n")
    print(final_df.head())
    print("-" * 50)

    print("\nBirleştirilmiş DataFrame hakkında genel bilgi (final_df.info()):\n")
    final_df.info()
    print("-" * 50)

    # Birleştirilmiş DataFrame'i nihai eğitim veri seti olarak kaydet
    print(f"\nNihai eğitim veri seti '{output_file_path}' olarak kaydediliyor...")
    final_df.to_csv(output_file_path, index=False)
    print("Kaydetme tamamlandı.")

except FileNotFoundError as e:
    print(f"Hata: Dosya bulunamadı - {e.filename}. Lütfen dosya yollarını kontrol edin.")
except Exception as e:
    print(f"Bir hata oluştu: {e}")
