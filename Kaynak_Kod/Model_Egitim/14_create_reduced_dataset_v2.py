# 14_create_reduced_dataset_v2.py
import pandas as pd
import os

print("--- Yeni Azaltılmış Veri Seti Oluşturuluyor (v2) ---")

# Dosya yollarını tanımla
current_dir = os.getcwd()
input_file_path = os.path.join(current_dir, 'model_ready_dataset.csv')
output_file_path = os.path.join(current_dir, 'model_ready_reduced_dataset_v2.csv')

try:
    # Modellenmeye hazır veri setini yükle
    print(f"'{input_file_path}' yükleniyor...")
    df = pd.read_csv(input_file_path)
    print(f"Orijinal DataFrame boyutu: {df.shape}")

    # Kaldırılacak sütunları bul
    cols_to_drop = [col for col in df.columns if col.startswith('city_') or col.startswith('gender_')]
    
    # Sütunları kaldır
    df_reduced = df.drop(columns=cols_to_drop)
    print(f"- {len(cols_to_drop)} adet sütun kaldırıldı.")
    print(f"Yeni DataFrame boyutu: {df_reduced.shape}")

    # Yeni veri setini kaydet
    print(f"\nAzaltılmış veri seti '{output_file_path}' olarak kaydediliyor...")
    df_reduced.to_csv(output_file_path, index=False)
    print("Kaydetme tamamlandı.")

except FileNotFoundError as e:
    print(f"Hata: Dosya bulunamadı - {e.filename}. Lütfen dosya yollarını kontrol edin.")
except Exception as e:
    print(f"Bir hata oluştu: {e}")
