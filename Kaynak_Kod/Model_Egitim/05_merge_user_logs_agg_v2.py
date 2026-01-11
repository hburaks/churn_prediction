# 05_merge_user_logs_agg_v2.py
import pandas as pd
import os
import numpy as np

print("--- user_logs_aggregated_v2.csv ile ana veri seti birleştiriliyor ---")

# Dosya yollarını tanımla
current_dir = os.getcwd()
train_members_transactions_merged_path = os.path.join(current_dir, 'train_members_transactions_merged.csv')
user_logs_aggregated_v2_path = os.path.join(current_dir, 'user_logs_aggregated_v2.csv')
output_file_path = os.path.join(current_dir, 'final_train_dataset.csv')

try:
    # train_members_transactions_merged.csv dosyasını yükle
    print("'{}' yükleniyor...".format(train_members_transactions_merged_path))
    df_merged = pd.read_csv(train_members_transactions_merged_path)
    print("Yüklenen DataFrame boyutu: {}".format(df_merged.shape))

    # user_logs_aggregated_v2.csv dosyasını yükle
    print("'{}' yükleniyor...".format(user_logs_aggregated_v2_path))
    df_user_logs_agg = pd.read_csv(user_logs_aggregated_v2_path)
    print("Yüklenen user_logs_aggregated_v2 DataFrame boyutu: {}".format(df_user_logs_agg.shape))

    # msno sütunu üzerinden sol birleştirme yap
    print("\nmsno' sütunu üzerinden sol birleştirme yapılıyor...")
    final_df = pd.merge(df_merged, df_user_logs_agg, on='msno', how='left')
    print("Birleştirme sonrası DataFrame boyutu: {}".format(final_df.shape))

    # Birleştirme sonrası oluşan NaN değerlerini doldur
    user_logs_cols = [col for col in final_df.columns if col.startswith(('num_', 'total_secs_')) and col not in df_merged.columns]
    print("\nBirleştirme sonrası oluşan NaN değerleri (user_logs sütunları için) 0 ile dolduruluyor...")
    final_df[user_logs_cols] = final_df[user_logs_cols].fillna(0)
    
    print("\nNihai DataFrame'in ilk 5 satırı:\n")
    print(final_df.head())
    print("-" * 50)

    print("\nNihai DataFrame hakkında genel bilgi:\n")
    final_df.info()
    print("-" * 50)

    # Nihai DataFrame'i kaydet
    print("\nNihai DataFrame '{}' olarak kaydediliyor...".format(output_file_path))
    final_df.to_csv(output_file_path, index=False)
    print("Kaydetme tamamlandı.")

except FileNotFoundError as e:
    print("Hata: Dosya bulunamadı - {}".format(e.filename))
except Exception as e:
    print("Bir hata oluştu: {}".format(e))
