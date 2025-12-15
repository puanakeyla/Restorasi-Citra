import cv2
import numpy as np
from pathlib import Path

IMAGES = [
    {
        "caption": "Gambar Landscape Salt & Pepper Level 2",
        "filename": "pemandanganLS.jpg",
    },
    {
        "caption": "Gambar Landscape Grayscale Salt & Pepper Level 2",
        "filename": "pemandanganLS_gray.jpg",
    },
    {
        "caption": "Gambar Potrait Salt & Pepper Level 2",
        "filename": "objekPT.jpg",
    },
    {
        "caption": "Gambar Potrait Grayscale Salt & Pepper Level 2",
        "filename": "objekPT_gray.jpg",
    },
]

NOISE_AMOUNT = 0.06  # Level 2 lebih kuat dari level 1
SALT_VS_PEPPER = 0.5
SHOW_WINDOWS = True
MAX_DISPLAY_WIDTH = 1280
MAX_DISPLAY_HEIGHT = 720


def add_salt_pepper_noise(image: np.ndarray, amount: float, salt_vs_pepper: float) -> np.ndarray:
    noisy = image.copy()
    height, width = image.shape[:2]
    total_pixels = height * width

    num_salt = int(np.ceil(amount * total_pixels * salt_vs_pepper))
    num_pepper = int(np.ceil(amount * total_pixels * (1.0 - salt_vs_pepper)))

    coords_salt = (
        np.random.randint(0, height, num_salt),
        np.random.randint(0, width, num_salt),
    )
    coords_pepper = (
        np.random.randint(0, height, num_pepper),
        np.random.randint(0, width, num_pepper),
    )

    if noisy.ndim == 2:
        noisy[coords_salt] = 255
        noisy[coords_pepper] = 0
    else:
        noisy[coords_salt[0], coords_salt[1], :] = 255
        noisy[coords_pepper[0], coords_pepper[1], :] = 0

    return noisy


def _resize_for_display(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    width_scale = MAX_DISPLAY_WIDTH / width
    height_scale = MAX_DISPLAY_HEIGHT / height
    scale = min(width_scale, height_scale, 1.0)
    if scale == 1.0:
        return image
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def main():
    base_dir = Path(__file__).resolve().parent

    for entry in IMAGES:
        caption = entry["caption"]
        filename = entry["filename"]
        image_path = base_dir / filename

        if not image_path.exists():
            print(f"Lewati {filename}: file tidak ditemukan")
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"Lewati {filename}: gagal dibuka")
            continue

        if image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        noisy = add_salt_pepper_noise(image, NOISE_AMOUNT, SALT_VS_PEPPER)

        safe_caption = caption.replace(" ", "_").replace("&", "dan")
        output_path = base_dir / f"{safe_caption}.jpg"
        cv2.imwrite(str(output_path), noisy)

        comparison_original = image
        if comparison_original.ndim == 2:
            comparison_original = cv2.cvtColor(comparison_original, cv2.COLOR_GRAY2BGR)
        comparison_noisy = noisy
        if comparison_noisy.ndim == 2:
            comparison_noisy = cv2.cvtColor(comparison_noisy, cv2.COLOR_GRAY2BGR)

        comparison = np.hstack((comparison_original, comparison_noisy))
        comparison_path = base_dir / f"{safe_caption}_comparison.jpg"
        cv2.imwrite(str(comparison_path), comparison)

        print(f"{caption} disimpan ke {output_path.name}")
        print(f"Perbandingan disimpan ke {comparison_path.name}")

        if SHOW_WINDOWS:
            cv2.imshow(f"Original - {caption}", _resize_for_display(comparison_original))
            cv2.imshow(f"Noise Level 2 - {caption}", _resize_for_display(comparison_noisy))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    print("Noise Salt & Pepper Level 2 selesai dibangkitkan untuk seluruh citra.")


if __name__ == "__main__":
    main()
