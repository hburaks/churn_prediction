# 02_aggregate_transactions.py
import pandas as pd
import os

print("--- transactions.csv dosyası özetleniyor (aggregation) ---")

# Dosya yollarını tanımla
current_dir = os.getcwd()
data_dir = os.path.join(current_dir, 'kkbox-churn-prediction-challenge')
transactions_file_path = os.path.join(data_dir, 'transactions.csv')
output_file_path = os.path.join(current_dir, 'transactions_aggregated.csv')

try:
    # transactions.csv dosyasını yükle
    print(f"'{transactions_file_path}' yükleniyor...")
    transactions_df = pd.read_csv(transactions_file_path)
    print(f"transactions_df boyutu: {transactions_df.shape}")

    # Yeni özellik: indirim (discount)
    # Kullanıcının listedeki fiyattan ne kadar indirimli ödediğini gösterir.
    transactions_df['discount'] = transactions_df['plan_list_price'] - transactions_df['actual_amount_paid']

    # msno'ya göre grupla ve özet istatistikleri hesapla
    # .agg() fonksiyonu, her bir sütun için farklı bir veya birden fazla işlem yapmamızı sağlar.
    print("Kullanıcı bazında özet istatistikler hesaplanıyor...")
    
    # Yapılacak aggregation işlemlerini bir sözlükte tanımla
    aggregations = {
        'payment_plan_days': ['mean', 'sum'],
        'plan_list_price': ['mean', 'sum'],
        'actual_amount_paid': ['mean', 'sum'],
        'is_auto_renew': ['max'], # Kullanıcı hiç otomatik yenileme kullandı mı? (1 evet, 0 hayır)
        'is_cancel': ['sum'], # Toplam iptal edilen işlem sayısı
        'discount': ['mean', 'sum'],
        'msno': ['count'] # Toplam işlem sayısı
    }

    # Gruplama ve aggregation işlemini yap
    transactions_agg_df = transactions_df.groupby('msno').agg(aggregations)

    # Sütun isimlerini daha anlaşılır hale getir
    # Örn: ('payment_plan_days', 'mean') -> 'payment_plan_days_mean'
    transactions_agg_df.columns = ['_'.join(col).strip() for col in transactions_agg_df.columns.values]
    transactions_agg_df.rename(columns={'msno_count': 'total_transactions'}, inplace=True)

    print("\\nÖzetlenmiş DataFrame'in ilk 5 satırı (transactions_agg_df.head()):\\n")
    print(transactions_agg_df.head())
    print("-" * 50)

    print("\\nÖzetlenmiş DataFrame hakkında genel bilgi (transactions_agg_df.info()):\\n")
    transactions_agg_df.info()
    print("-" * 50)

    # Özetlenmiş DataFrame'i yeni bir CSV dosyasına kaydet
    print(f"\\nÖzetlenmiş DataFrame '{output_file_path}' olarak kaydediliyor...")
    transactions_agg_df.to_csv(output_file_path) # index=True çünkü index'te msno bilgisi var
    print("Kaydetme tamamlandı.")

except FileNotFoundError as e:
    print(f"Hata: Dosya bulunamadı - {e.filename}. Lütfen dosya yollarını kontrol edin.")
except Exception as e:
    print(f"Bir hata oluştu: {e}")
