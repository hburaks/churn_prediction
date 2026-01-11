# 08_feature_importance_logreg.py
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import warnings

# Olası convergence uyarılarını bastır
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

print("--- Lojistik Regresyon ile Özellik Önemini Belirleme ---")

# Dosya yolunu tanımla
current_dir = os.getcwd()
input_file_path = os.path.join(current_dir, 'model_ready_dataset.csv')

try:
    # Modellenmeye hazır veri setini yükle
    print(f"'{input_file_path}' yükleniyor...")
    df = pd.read_csv(input_file_path)

    # 1. Özellik (X) ve Hedef (y) Değişkenlerini Ayırma
    X = df.drop('is_churn', axis=1)
    y = df['is_churn']
    
    # Özellik isimlerini daha sonra kullanmak için sakla
    feature_names = X.columns

    # 2. Veri Setini Eğitim ve Test Olarak Ayırma
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. Özellikleri Ölçeklendirme
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # 4. Lojistik Regresyon Modelini Eğitme
    print("\nModel eğitiliyor...")
    model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    print("- Model başarıyla eğitildi.")

    # 5. Model Katsayılarını (Coefficients) Çekme ve Analiz Etme
    print("\n--- Özellik Önem Dereceleri (Katsayılar) ---")
    
    # Katsayıları ve özellik isimlerini bir DataFrame'de birleştir
    coefficients = model.coef_[0]
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
    
    # Katsayıların mutlak değerine göre sıralama yapmak için yeni bir sütun ekle
    feature_importance_df['Absolute_Coefficient'] = feature_importance_df['Coefficient'].abs()
    
    # Mutlak katsayıya göre büyükten küçüğe sırala
    feature_importance_df = feature_importance_df.sort_values(by='Absolute_Coefficient', ascending=False)

    print("\nEn Etkili 15 Özellik (Pozitif veya Negatif):\n")
    print(feature_importance_df.head(15).to_string())

    print("\nEn Etkisiz 15 Özellik (Tahmine Etkisi En Az Olanlar):\n")
    print(feature_importance_df.tail(15).to_string())
    
    print("\n--------------------------------------------------")
    print("Yorum:")
    print("- Pozitif katsayı, o özelliğin artmasının churn olasılığını artırdığını gösterir.")
    print("- Negatif katsayı, o özelliğin artmasının churn olasılığını azalttığını (müşterinin kalma olasılığını artırdığını) gösterir.")
    print("- 'En Etkisiz Özellikler' listesindekiler, katsayıları sıfıra en yakın olanlardır ve modelden çıkarılabilecek adaylardır.")


except FileNotFoundError as e:
    print(f"Hata: Dosya bulunamadı - {e.filename}. Lütfen dosya yollarını kontrol edin.")
except Exception as e:
    print(f"Bir hata oluştu: {e}")
