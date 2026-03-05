import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

CONTOH_GAMBAR = "dataset/testing/mawar_putih/P25.jpg" 
FILE_TRAINING = "=data_testing.xlsx"

def visualisasi_preprocessing():

    if not os.path.exists(CONTOH_GAMBAR):
        print(f"ERROR: File gambar contoh tidak ditemukan: {CONTOH_GAMBAR}")
        print("Ganti path di variabel 'CONTOH_GAMBAR' dulu!")
        return

    img = cv2.imread(CONTOH_GAMBAR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    
    img_resized = cv2.resize(img, (300, 300))
    img_resized_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask_daun = cv2.inRange(hsv, lower_green, upper_green)
    mask_bukan_daun = cv2.bitwise_not(mask_daun)
    
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    final_mask = cv2.bitwise_and(mask_otsu, mask_otsu, mask=mask_bukan_daun)
    
    hasil_segmentasi = cv2.bitwise_and(img_resized_rgb, img_resized_rgb, mask=final_mask)

    plt.figure(figsize=(12, 8))
    
    
    plt.subplot(2, 3, 1)
    plt.imshow(img_rgb)
    plt.title("1. Citra Asli (Raw)")
    plt.axis('off')
    
    
    plt.subplot(2, 3, 2)
    plt.imshow(img_resized_rgb)
    plt.title("2. Resize (300x300)")
    plt.axis('off')
    
    
    plt.subplot(2, 3, 3)
    plt.imshow(mask_bukan_daun, cmap='gray')
    plt.title("3. Filter Warna (Hapus Daun)")
    plt.axis('off')

    
    plt.subplot(2, 3, 4)
    plt.imshow(mask_otsu, cmap='gray')
    plt.title("4. Otsu Thresholding")
    plt.axis('off')
    
    
    plt.subplot(2, 3, 5)
    plt.imshow(final_mask, cmap='gray')
    plt.title("5. Masker Final (ROI)")
    plt.axis('off')

    
    plt.subplot(2, 3, 6)
    plt.imshow(hasil_segmentasi)
    plt.title("6. Hasil Segmentasi")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    print("-> Gambar Preprocessing muncul! Silakan Screenshot/Save.")

def visualisasi_scatter_plot():
    print("\nSedang membuat Scatter Plot Data...")
    
    if not os.path.exists(FILE_TRAINING):
        print("ERROR: File Excel data_training.xlsx tidak ditemukan!")
        return

    df = pd.read_excel(FILE_TRAINING)
    
    merah = df[df['Label_Asli'] == 'mawar_merah']
    putih = df[df['Label_Asli'] == 'mawar_putih']
    kuning = df[df['Label_Asli'] == 'mawar_kuning']
    
    plt.figure(figsize=(10, 6))

    plt.scatter(merah['Red'], merah['Green'], color='red', label='Mawar Merah', alpha=0.6)
    plt.scatter(putih['Red'], putih['Green'], color='gray', label='Mawar Putih', alpha=0.6)
    plt.scatter(kuning['Red'], kuning['Green'], color='gold', label='Mawar Kuning', alpha=0.6)
    
    plt.title("Distribusi Data Fitur Warna (Red vs Green)", fontsize=14, fontweight='bold')
    plt.xlabel("Nilai Rata-rata Kanal Red (R)", fontsize=12)
    plt.ylabel("Nilai Rata-rata Kanal Green (G)", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.show()
    print("-> Grafik Scatter Plot muncul! Silakan Screenshot/Save.")

if __name__ == "__main__":
    visualisasi_preprocessing()
    visualisasi_scatter_plot()