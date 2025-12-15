import cv2
import numpy as np
import os

# ===== KONFIGURASI =====
PATH_FILE = './input/salt-pepper/landscape-rgb-saltpepper-2.jpg'
# hapus ekstensi file
NAME_FILE = os.path.splitext(os.path.basename(PATH_FILE))[0]
OUTPUT_FILE = f'{NAME_FILE}-min.jpg'
# =======================

# Baca gambar
img = cv2.imread(PATH_FILE, cv2.IMREAD_UNCHANGED)
if img is None:
    print("Error: Gambar tidak ditemukan!")
    exit()

# Deteksi otomatis apakah gambar RGB atau Grayscale dari channel
if len(img.shape) == 3:
    USE_RGB = True
    height, width, channels = img.shape
    output = np.zeros((height, width, channels), dtype=np.uint8)
else:
    USE_RGB = False
    height, width = img.shape
    output = np.zeros((height, width), dtype=np.uint8)

# Min Filter 3x3 dengan manual pixel manipulation
if USE_RGB:
    # Proses untuk gambar RGB (3 channel)
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            # Proses setiap channel (B, G, R)
            for c in range(channels):
                pixel_values = []
                
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        pixel_values.append(int(img[i + di, j + dj, c]))
                
                # Cari nilai minimum dari 9 pixel
                min_value = pixel_values[0]
                for val in pixel_values:
                    if val < min_value:
                        min_value = val
                
                output[i, j, c] = min_value
    
    # Handle border pixels (copy dari gambar asli)
    output[0, :, :] = img[0, :, :]
    output[height-1, :, :] = img[height-1, :, :]
    output[:, 0, :] = img[:, 0, :]
    output[:, width-1, :] = img[:, width-1, :]
else:
    # Proses untuk gambar Grayscale
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            pixel_values = []
            
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    pixel_values.append(int(img[i + di, j + dj]))
            
            # Cari nilai minimum dari 9 pixel
            min_value = pixel_values[0]
            for val in pixel_values:
                if val < min_value:
                    min_value = val
            
            output[i, j] = min_value
    
    # Handle border pixels (copy dari gambar asli)
    output[0, :] = img[0, :]
    output[height-1, :] = img[height-1, :]
    output[:, 0] = img[:, 0]
    output[:, width-1] = img[:, width-1]

# Buat folder output jika belum ada
if not os.path.exists('output'):
    os.makedirs('output')

# Simpan hasil
cv2.imwrite(f'output/salt-pepper/{OUTPUT_FILE}', output)

print("Filter berhasil diterapkan!")
print(f"Mode: {'RGB' if USE_RGB else 'Grayscale'} (Terdeteksi otomatis)")
print(f"Hasil disimpan di: output/salt-pepper/{OUTPUT_FILE}")