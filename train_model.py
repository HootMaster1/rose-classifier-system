import pandas as pd
import numpy as np
import joblib 
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay

FILE_TRAIN = "data_training.xlsx"
FILE_TEST  = "data_testing.xlsx"
OUTPUT_MODEL = "model_knn.pkl"
OUTPUT_SCALER = "scaler.pkl"

def main():
    print("=== TRAINING MODEL TESTING ===")

    try:
        df_train = pd.read_excel(FILE_TRAIN)
        df_test = pd.read_excel(FILE_TEST)
    except FileNotFoundError:
        print("ERROR: File Excel tidak ada. Run generate_dataset.py dulu.")
        return

    X_train = df_train.drop(["Label_Asli", "Nama_File"], axis=1, errors='ignore')
    y_train = df_train["Label_Asli"]
    
    X_test = df_test.drop(["Label_Asli", "Nama_File", "Prediksi_Sistem", "Status"], axis=1, errors='ignore')
    y_test = df_test["Label_Asli"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train) 
    X_test_scaled = scaler.transform(X_test)       

    k = 7
    print(f"-> Training KNN (K={k})...")
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)

    y_pred = knn.predict(X_test_scaled)
    akurasi = accuracy_score(y_test, y_pred)

    print(f"\nAKURASI AKHIR: {akurasi * 100:.2f}%")
    
    print("\n--- DETAIL LAPORAN KLASIFIKASI ---")
    print(classification_report(y_test, y_pred))

    joblib.dump(knn, OUTPUT_MODEL)
    joblib.dump(scaler, OUTPUT_SCALER)
    print("Model disimpan.")

    try:
        print("-> Membuat Confusion Matrix...")
        cm = confusion_matrix(y_test, y_pred)
        
        labels_urut = sorted(knn.classes_)
        
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_urut)

        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(cmap='Blues', ax=ax, values_format='d') 
        
        plt.title(f"Confusion Matrix (Akurasi: {akurasi*100:.2f}%)", fontsize=14, pad=20)
        plt.ylabel('Label Asli (Fakta)', fontsize=12)
        plt.xlabel('Prediksi Sistem', fontsize=12)
        plt.grid(False) 
        
        nama_file_cm = "hasil_confusion_matrix.png"
        plt.savefig(nama_file_cm, dpi=300, bbox_inches='tight')
        print(f"SUKSES: Grafik Confusion Matrix disimpan sebagai '{nama_file_cm}'")
        
        plt.close() 
        
    except Exception as e:
        print(f"Info: Gagal membuat grafik matrix ({e})")

if __name__ == "__main__":
    main()