import cv2
import os
import numpy as np


TARGET_DIRS = [
    "dataset/testing/mawar_merah,"
    "dataset/testing/mawar_kuning,"
    "dataset/testing/mawar_putih"
]

def augment_image(img):
    generated_images = []
    
    generated_images.append(img)

    generated_images.append(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))
    
    generated_images.append(cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE))
    
    generated_images.append(cv2.flip(img, 1))
    
    generated_images.append(cv2.flip(img, 0))
    
    matrix_bright = np.ones(img.shape, dtype="uint8") * 35
    generated_images.append(cv2.add(img, matrix_bright))

    return generated_images

def main():
    print("=== MEMULAI AUGMENTASI DATA (PERBANYAK OTOMATIS) ===")
    
    for folder in TARGET_DIRS:

        if not os.path.exists(folder):
            print(f"ERROR: Folder tidak ditemukan -> {folder}")
            print("Pastikan kamu sudah bikin foldernya dan isi minimal 1 foto!")
            continue
            
        print(f"\n-> Memproses folder: {folder}...")
        
        files = os.listdir(folder)
        original_count = 0
        new_count = 0
        
        for filename in files:
    
            if filename.startswith("aug_"):
                continue
                
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            file_path = os.path.join(folder, filename)
            img = cv2.imread(file_path)
            
            if img is None:
                continue
            
            original_count += 1

            variasi_list = augment_image(img)

            base_name = os.path.splitext(filename)[0]
            for i, variasi in enumerate(variasi_list):
                if i == 0: continue 

                new_filename = f"aug_{base_name}_{i}.jpg"
                save_path = os.path.join(folder, new_filename)
                cv2.imwrite(save_path, variasi)
                new_count += 1
        
        print(f"   Status: {original_count} foto asli -> Menghasilkan {new_count} foto baru.")
        print(f"   Total isi folder sekarang: {original_count + new_count} file.")

    print("\n=== SELESAI. Silakan lanjut ke generate_dataset.py ===")

if __name__ == "__main__":
    main()