"""
utils/video.py
--------------
Utilitas ekstraksi frame dari video.
Fungsi utama:
  - extract_16_frames(video_path)      → list of BGR numpy arrays
  - prepare_cropped_frames(...)        → list of PIL.Image
"""

import os
import glob
import cv2
from PIL import Image

from core.face_detector import crop_face


def extract_16_frames(video_path: str) -> list:
    """Ambil 16 frame yang terdistribusi merata dari video."""
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release()
        return []

    indices = [int(total * i / 16) for i in range(16)]
    frames  = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames


def prepare_cropped_frames(
    video_path: str,
    root_folder: str,
    crop_dir_base: str,
) -> tuple:
    """
    Ekstrak, crop wajah, resize, simpan ke disk, dan kembalikan sebagai list PIL.Image.
    Jika 16 frame sudah ada di disk, langsung load tanpa re-process.

    Returns:
        (pil_images: list, no_face_count: int)
        no_face_count = jumlah frame yang tidak terdeteksi wajahnya oleh MediaPipe.
        Jika loading dari cache (frame sudah ada), no_face_count = 0 (tidak bisa diketahui).
    """
    rel_path        = os.path.relpath(video_path, root_folder)
    base_name       = os.path.splitext(rel_path)[0]
    target_crop_dir = os.path.join(crop_dir_base, base_name)
    os.makedirs(target_crop_dir, exist_ok=True)

    saved_files = sorted(glob.glob(os.path.join(target_crop_dir, "frame_*.jpg")))
    if len(saved_files) == 16:
        return [Image.open(f).convert("RGB") for f in saved_files], 0

    frames_bgr = extract_16_frames(video_path)
    if not frames_bgr:
        return [], 0

    pil_images    = []
    no_face_count = 0
    for i, frame in enumerate(frames_bgr):
        cropped_sq, face_found = crop_face(frame)
        if not face_found:
            no_face_count += 1
        resized_full = cv2.resize(cropped_sq, (512, 512), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(target_crop_dir, f"frame_{i:02d}.jpg"), resized_full)
        pil_images.append(
            Image.fromarray(cv2.cvtColor(resized_full, cv2.COLOR_BGR2RGB))
        )
    return pil_images, no_face_count

