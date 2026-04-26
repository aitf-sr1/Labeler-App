"""
utils/io.py

Fungsi baca/tulis untuk semua file persistensi data anotasi.
Path dioper dari luar (VideoLabelerApp) — modul ini tidak menyimpan state apapun.

Alur umum:
    open_folder()        -> _load_data() memanggil semua load_*()
    save_current_state() -> save_annotations() + save_flagged() + save_frame_annotations()
    skip_video()         -> save_skipped()
    _apply_siglip_result() -> save_batch_history()
"""

import os
import csv
import json


def load_annotations(csv_path: str) -> dict:
    """Baca annotations_bener.csv. Return {rel_path: [boredom, engagement, confusion, frustration]}."""
    data = {}
    if not os.path.exists(csv_path):
        return data
    with open(csv_path, newline="") as f:
        for row in list(csv.reader(f))[1:]:
            if len(row) >= 8:
                data[row[3]] = row[4:8]
    return data


def save_annotations(csv_path: str, annotations_data: dict) -> None:
    """Tulis ulang annotations_bener.csv dari dict. Dipanggil setiap kali Save & Next."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["UUID", "Video_Asli", "Clip_Name", "File_Path",
                    "Boredom", "Engagement", "Confusion", "Frustration"])
        # Gunakan list() untuk menghindari RuntimeError jika dict berubah di thread lain
        for rel, vals in list(annotations_data.items()):
            p = rel.split(os.sep)
            w.writerow([
                p[0] if p else "-",
                p[1] if len(p) > 1 else "-",
                p[-1],
                rel,
            ] + list(vals))


def load_flagged(csv_path: str) -> set:
    """Baca flagged_videos.csv. Return set rel_path yang di-flag."""
    flagged = set()
    if not os.path.exists(csv_path):
        return flagged
    with open(csv_path, newline="") as f:
        for row in list(csv.reader(f))[1:]:
            if len(row) >= 4:
                flagged.add(row[3])
    return flagged


def save_flagged(csv_path: str, flagged_data: set) -> None:
    """Tulis ulang flagged_videos.csv dari set rel_path. Video yang masuk sini dikecualikan dari training."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["UUID", "Video_Asli", "Clip_Name", "File_Path"])
        for rel in sorted(flagged_data):
            p = rel.split(os.sep)
            w.writerow([
                p[0] if p else "-",
                p[1] if len(p) > 1 else "-",
                p[-1],
                rel,
            ])


def load_frame_annotations(json_path: str) -> dict:
    """Baca frame_annotations.json. Return {rel_path: {frame_idx: {label: 0|1}}}."""
    if not os.path.exists(json_path):
        return {}
    with open(json_path, "r") as f:
        return json.load(f)


def save_frame_annotations(json_path: str, frame_annotations: dict) -> None:
    """Tulis ulang frame_annotations.json."""
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w") as f:
        # Gunakan dict() untuk copy guna menghindari RuntimeError
        json.dump(dict(frame_annotations), f, indent=4)


def load_batch_history(json_path: str) -> dict:
    """
    Baca batch_history.json — riwayat hasil inferensi AI per video.
    Digunakan untuk skip otomatis pada video yang sudah diproses.
    Return {} jika file belum ada atau rusak.
    """
    if not os.path.exists(json_path):
        return {}
    try:
        with open(json_path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def save_batch_history(json_path: str, batch_history: dict) -> None:
    """Tulis ulang batch_history.json. Dipanggil setelah setiap video selesai diproses AI."""
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    try:
        with open(json_path, "w") as f:
            # Gunakan dict() untuk copy guna menghindari RuntimeError
            json.dump(dict(batch_history), f, indent=4)
    except Exception as e:
        print(f"Gagal menyimpan batch_history: {e}")


def load_skipped(json_path: str) -> set:
    """Baca skipped_videos.json. Return set rel_path yang di-skip."""
    if not os.path.exists(json_path):
        return set()
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
            return set(data) if isinstance(data, list) else set()
    except Exception:
        return set()


def save_skipped(json_path: str, skipped_videos: set) -> None:
    """Tulis ulang skipped_videos.json. Dipanggil setiap kali user klik Skip."""
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    try:
        with open(json_path, "w") as f:
            json.dump(list(skipped_videos), f, indent=4)
    except Exception as e:
        print(f"Gagal menyimpan skipped_videos: {e}")


def load_thresholds(json_path: str, labels: list) -> list | None:
    """
    Baca thresholds.json. Return list float per label, atau None jika belum ada.

    Format file: {"Boredom": 0.45, "Engagement": 0.50, ...}
    """
    if not os.path.exists(json_path):
        return None
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        return [data.get(lbl, 0.5) for lbl in labels]
    except Exception:
        return None


def save_thresholds(json_path: str, labels: list, thresholds: list) -> None:
    """Simpan threshold per label ke thresholds.json. Dipanggil saat threshold diubah."""
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    data = {lbl: round(thr, 2) for lbl, thr in zip(labels, thresholds)}
    try:
        with open(json_path, "w") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"Gagal menyimpan thresholds: {e}")
