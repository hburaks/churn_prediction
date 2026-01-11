import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import time

print("--- MLP Model Eğitimi Başlatılıyor ---")

# Dosya yolları
current_dir = os.getcwd()
X_train_path = os.path.join(current_dir, 'mlp_X_train.npy')
X_test_path = os.path.join(current_dir, 'mlp_X_test.npy')
y_train_path = os.path.join(current_dir, 'mlp_y_train.npy')
y_test_path = os.path.join(current_dir, 'mlp_y_test.npy')
model_out = os.path.join(current_dir, 'mlp_model.keras')
results_out = os.path.join(current_dir, 'mlp_results.md')

try:
    # Verileri yükle
    print("İşlenmiş veriler yükleniyor...")
    X_train = np.load(X_train_path)
    X_test = np.load(X_test_path)
    y_train = np.load(y_train_path)
    y_test = np.load(y_test_path)
    print(f"Eğitim seti: {X_train.shape}, Test seti: {X_test.shape}")

    # Model mimarisini tanımla
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # Modeli derle
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

    # Early Stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Modeli eğit
    print("\nModel eğitiliyor...")
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=1024,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )
    end_time = time.time()
    print(f"\nEğitim tamamlandı. Süre: {end_time - start_time:.2f} saniye")

    # Modeli kaydet
    model.save(model_out)
    print(f"Model '{model_out}' olarak kaydedildi.")

    # Test seti üzerinde değerlendirme
    print("\nTest seti üzerinde tahminler yapılıyor...")
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob >= 0.5).astype(int)

    # Sonuçları raporla
    report = classification_report(y_test, y_pred, target_names=['Not Churn (0)', 'Churn (1)'])
    cm = confusion_matrix(y_test, y_pred)

    print("\n--- MLP Model Sonuçları ---")
    print(report)

    # Sonuçları dosyaya kaydet
    with open(results_out, 'w') as f:
        f.write("# MLP Model Sonuçları\n\n")
        f.write(f"Eğitim Süresi: {end_time - start_time:.2f} saniye\n\n")
        f.write("## Sınıflandırma Raporu\n")
        f.write(f"```\n{report}\n```\n\n")
        f.write("## Karışıklık Matrisi\n")
        f.write(f"```\n{cm}\n```\n")

except Exception as e:
    print(f"Bir hata oluştu: {e}")
