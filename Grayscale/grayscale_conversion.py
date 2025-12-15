import cv2
import numpy as np
from pathlib import Path

IMAGES_TO_PROCESS = ["pemandanganLS.jpg", "objekPT.jpg"]
SHOW_WINDOWS = True  # set False jika tidak ingin membuka tampilan
MAX_DISPLAY_WIDTH = 1280
MAX_DISPLAY_HEIGHT = 720

def _resize_for_display(image: np.ndarray) -> np.ndarray:
    """Scale image proporsional agar tidak terpotong di window."""
    height, width = image.shape[:2]
    width_scale = MAX_DISPLAY_WIDTH / width
    height_scale = MAX_DISPLAY_HEIGHT / height
    scale = min(width_scale, height_scale, 1.0)
    if scale == 1.0:
        return image
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

def convert_to_grayscale(image_path: Path, output_dir: Path) -> Path:
    """Convert the given color image to grayscale, save, dan tampilkan hasil."""
    color = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if color is None:
        raise FileNotFoundError(f"Gagal memuat gambar: {image_path}")

    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    output_path = output_dir / f"{image_path.stem}_gray.jpg"
    cv2.imwrite(str(output_path), gray)

    comparison = np.hstack((color, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)))
    preview_path = output_dir / f"{image_path.stem}_comparison.jpg"
    cv2.imwrite(str(preview_path), comparison)

    if SHOW_WINDOWS:
        cv2.namedWindow(f"Grayscale - {image_path.name}", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow(f"Before vs After - {image_path.name}", cv2.WINDOW_AUTOSIZE)
        cv2.imshow(
            f"Grayscale - {image_path.name}",
            _resize_for_display(gray)
        )
        cv2.imshow(
            f"Before vs After - {image_path.name}",
            _resize_for_display(comparison)
        )
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(f"Berhasil simpan grayscale: {output_path}")
    print(f"Berhasil simpan perbandingan: {preview_path}")
    return output_path


def main():
    work_dir = Path(__file__).resolve().parent
    missing_files = []

    for image_name in IMAGES_TO_PROCESS:
        image_path = work_dir / image_name
        if not image_path.exists():
            missing_files.append(image_name)
            continue
        convert_to_grayscale(image_path, work_dir)

    if missing_files:
        print("File berikut tidak ditemukan dan dilewati:")
        for name in missing_files:
            print(f" - {name}")

    print("Konversi selesai untuk semua gambar yang tersedia.")


if __name__ == "__main__":
    main()
