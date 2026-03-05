import cv2
import numpy as np
import os
import pandas as pd
import joblib

BASE_DIR = "dataset"
CLASSES = ["mawar_merah", "mawar_putih", "mawar_kuning"] 

MODEL_PATH = "model_knn.pkl"
SCALER_PATH = "scaler.pkl"

def load_brain():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        print(f"[INFO] Model ditemukan! Mode 'Auto-Correction' AKTIF.")
        return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)
    else:
        print(f"[INFO] Model belum ada. Hanya generate data biasa.")
        return None, None

def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None: return None
    
    img_resized = cv2.resize(img, (300, 300))
    
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask_daun = cv2.inRange(hsv, lower_green, upper_green)
    mask_bukan_daun = cv2.bitwise_not(mask_daun)
    
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    final_mask = cv2.bitwise_and(mask_otsu, mask_otsu, mask=mask_bukan_daun)
    
    mean_val = cv2.mean(img_resized, mask=final_mask)[:3]
    hu = cv2.HuMoments(cv2.moments(final_mask)).flatten()
    hu_log = [-1 * np.sign(h) * np.log10(np.abs(h)) if h != 0 else 0 for h in hu]
    
    return [mean_val[2], mean_val[1], mean_val[0]] + hu_log

def process_folder(target_subfolder, output_filename, model=None, scaler=None):
    print(f"\n--- MEMPROSES FOLDER: {target_subfolder.upper()} ---")
    data_kolektif = []
    
    total_data = 0
    total_benar = 0
    
    for label_asli in CLASSES:
        folder_path = os.path.join(BASE_DIR, target_subfolder, label_asli)
        
        if not os.path.exists(folder_path):
            continue
            
        print(f"-> Mengambil data: {label_asli}...")
        files = os.listdir(folder_path)
        
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_path = os.path.join(folder_path, filename)
                features = extract_features(file_path)
                
                if features:
                    row_data = [filename] + features + [label_asli]

                    if target_subfolder == "testing" and model is not None:
                        feat_np = np.array([features]) 
                        feat_scaled = scaler.transform(feat_np)
                        prediksi = model.predict(feat_scaled)[0]
                        
                        is_correct = (prediksi == label_asli)
                        status = "BENAR" if is_correct else "SALAH"
                        
                        total_data += 1
                        if is_correct: total_benar += 1

                        row_data.append(prediksi)
                        row_data.append(status)
                    
                    data_kolektif.append(row_data)

    if len(data_kolektif) > 0:
        cols = ["Nama_File", "Red", "Green", "Blue", "Hu1", "Hu2", "Hu3", "Hu4", "Hu5", "Hu6", "Hu7", "Label_Asli"]
        
        if target_subfolder == "testing" and model is not None:
            cols.append("Prediksi_Sistem")
            cols.append("Status")
        
        df = pd.DataFrame(data_kolektif, columns=cols)
        df.to_excel(output_filename, index=False)
        print(f"SUKSES: {output_filename} dibuat.")
        
        if target_subfolder == "testing" and model is not None and total_data > 0:
            akurasi = (total_benar / total_data) * 100
            print(f"\n[RAPOR] Akurasi Testing: {akurasi:.2f}% ({total_benar}/{total_data} Benar)")
            
    else:
        print(f"GAGAL: Tidak ada data di {target_subfolder}.")

def main():
    knn_model, scaler_model = load_brain()
    process_folder("training", "data_training.xlsx", model=None, scaler=None)
    process_folder("testing", "data_testing.xlsx", model=knn_model, scaler=scaler_model)

if __name__ == "__main__":
    main()