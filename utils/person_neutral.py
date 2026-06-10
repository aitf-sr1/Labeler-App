"""
utils/person_neutral.py

Kalibrasi baseline netral AU PER-ORANG (bukan populasi).

LATAR BELAKANG
--------------
FACS mendefinisikan intensitas Action Unit sebagai DEVIASI dari ekspresi NETRAL
(Bartlett 1999; Craig 2008). Bosch et al. (2023) menemukan ekspresi (frustrasi)
sangat bervariasi antar-individu dan merekomendasikan: "a universally trained
algorithm that is then customized towards each individual."

Baseline populasi (DEFAULT_AU_CALIB) mengabaikan ini — orang yang alisnya secara
struktural lebih rendah akan selalu salah-baca "confused". Modul ini menyimpan
nilai MediaPipe AU (baseline-normalized) saat tiap orang netral, lalu scoring
memakai nilai pribadi itu sebagai anchor neutral (lihat core/blendshape_features.py
compute_blendshape_features(..., person_neutral=...) ).

FORMAT person_neutrals.json (disimpan di folder dataset, sejajar batch_history.json):
    {
      "0b40c60f-4c96-4162-b26c-a7541ceeb34d": {
          "AU1": 0.00, "AU2": 0.00, "AU4": 0.05, "AU7": 0.10,
          "AU12": 0.00, "AU14": 0.01, "AU25": 0.02, "AU26": 0.03, "AU43": 0.08,
          "_video": "clips/.../emotion_part_01.mp4", "_frame": 0
      },
      ...
    }

Kunci = identitas orang (UUID dari path video).
Nilai AU = skor blendshape ter-normalisasi (compute_blendshape_features) saat netral.
Key "_video" dan "_frame" = lokasi frame acuan (untuk marker visual di galeri).
Key non-AU (underscore prefix) diabaikan saat scoring.
"""

import os
import json

_FNAME = "person_neutrals.json"
_cache = {}  # {dataset_dir: {uuid: {AU..}}}


def person_id_from_relpath(rel_path: str) -> str | None:
    """
    Ekstrak identitas orang dari rel_path video.
    Format: clips/data-batch-N-clips/{UUID}/{emotion}/{emotion}_part_XX.mp4
            → UUID (bagian ke-3).
    Fallback: kalau struktur beda, pakai segmen folder sebelum nama emosi.
    """
    if not rel_path:
        return None
    parts = rel_path.replace("\\", "/").split("/")
    # cari segmen UUID-like (panjang ~36 dgn '-') agar robust thd variasi kedalaman
    for p in parts:
        if len(p) >= 32 and p.count("-") >= 4:
            return p
    # fallback: 1 level di atas folder emosi (… /{person}/{emotion}/file.mp4)
    if len(parts) >= 3:
        return parts[-3]
    return None


def load_person_neutrals(dataset_dir: str) -> dict:
    """Baca person_neutrals.json dari folder dataset (cached). {} bila tak ada."""
    if not dataset_dir:
        return {}
    if dataset_dir in _cache:
        return _cache[dataset_dir]
    path = os.path.join(dataset_dir, _FNAME)
    data = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            data = {}
    _cache[dataset_dir] = data
    return data


def get_person_neutral(dataset_dir: str, rel_path: str) -> dict | None:
    """Baseline netral AU untuk orang pemilik rel_path, atau None bila belum ditandai."""
    uuid = person_id_from_relpath(rel_path)
    if not uuid:
        return None
    return load_person_neutrals(dataset_dir).get(uuid)


def set_person_neutral(dataset_dir: str, uuid: str, au_values: dict, meta: dict = None) -> str:
    """
    Simpan/timpa baseline netral satu orang ke person_neutrals.json.
    meta (opsional) = {"_video": rel_path, "_frame": idx} → lokasi frame yang dipilih
    sebagai acuan netral (untuk marker visual). Key non-AU diabaikan saat scoring.
    """
    if not dataset_dir or not uuid:
        return ""
    path = os.path.join(dataset_dir, _FNAME)
    data = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            data = {}
    entry = {k: round(float(v), 4) for k, v in au_values.items()}
    if meta:
        entry.update(meta)  # _video (str), _frame (int) — tidak di-float, diabaikan scoring
    data[uuid] = entry
    os.makedirs(dataset_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    _cache[dataset_dir] = data  # refresh cache
    print(f"[person_neutral] DISIMPAN: {os.path.abspath(path)} (total {len(data)} orang)")
    return os.path.abspath(path)


def get_person_neutral_frame(dataset_dir: str, rel_path: str):
    """
    Return frame_idx jika rel_path INI adalah video acuan netral orang tsb, else None.
    Dipakai untuk menggambar marker '★ NETRAL' di galeri.
    """
    n = get_person_neutral(dataset_dir, rel_path)
    if n and n.get("_video") == rel_path:
        return n.get("_frame")
    return None


def get_person_neutral_video(dataset_dir: str, rel_path: str):
    """Return rel_path video acuan netral untuk orang pemilik rel_path (untuk navigasi)."""
    n = get_person_neutral(dataset_dir, rel_path)
    return n.get("_video") if n else None


def invalidate_cache(dataset_dir: str = None) -> None:
    """Kosongkan cache (panggil setelah file diubah di luar set_person_neutral)."""
    if dataset_dir is None:
        _cache.clear()
    else:
        _cache.pop(dataset_dir, None)
