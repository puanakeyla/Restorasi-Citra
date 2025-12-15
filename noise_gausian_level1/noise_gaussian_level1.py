import cv2
import numpy as np
from pathlib import Path

IMAGES = [
    {
        "caption": "Gambar Landscape Gaussian Noise Level 1",
        "filename": "pemandanganLS.jpg",
    },
    {
        "caption": "Gambar Landscape Grayscale Gaussian Noise Level 1",
        "filename": "pemandanganLS_gray.jpg",
    },
    {
        "caption": "Gambar Potrait Gaussian Noise Level 1",
        "filename": "objekPT.jpg",
    },
    {
        "caption": "Gambar Potrait Grayscale Gaussian Noise Level 1",
        "filename": "objekPT_gray.jpg",
    },
]

NOISE_MEAN = 0.0  # Mean 0 menjaga terang keseluruhan tetap netral
NOISE_STDDEV = 12.0  # Level 1: noise ringan agar detail masih terlihat
SHOW_WINDOWS = True
MAX_DISPLAY_WIDTH = 1280
MAX_DISPLAY_HEIGHT = 720


def add_gaussian_noise(image: np.ndarray, mean: float, stddev: float) -> np.ndarray:
    image_float = image.astype(np.float32)
    noise = np.random.normal(mean, stddev, image.shape).astype(np.float32)
    noisy = image_float + noise
    noisy_clipped = np.clip(noisy, 0, 255)
    return noisy_clipped.astype(image.dtype)


def _resize_for_display(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    width_scale = MAX_DISPLAY_WIDTH / width
    height_scale = MAX_DISPLAY_HEIGHT / height
    scale = min(width_scale, height_scale, 1.0)
    if scale == 1.0:
        return image
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def main() -> None:
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

        noisy = add_gaussian_noise(image, NOISE_MEAN, NOISE_STDDEV)

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
            cv2.imshow(f"Gaussian Noise Level 1 - {caption}", _resize_for_display(comparison_noisy))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    print("Noise Gaussian Level 1 selesai dibangkitkan untuk seluruh citra.")


if __name__ == "__main__":
    main()
