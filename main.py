"""
main.py

Entry point untuk mode REST API menggunakan FastAPI.
Digunakan jika aplikasi dijalankan sebagai backend tanpa UI.

Cara menjalankan:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Endpoint:
    POST /analyze_video  -- inferensi SigLIP pada satu video
    GET  /health         -- cek apakah server berjalan
"""

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Tuple

from ai_service import prepare_cropped_frames, run_siglip_on_frames

app = FastAPI(title="SigLIP Video Labeling Microservice")


class InferenceRequest(BaseModel):
    """
    Body request untuk endpoint /analyze_video.

    Atribut:
        video_path:       Path absolut ke file video yang akan dianalisis.
        root_folder:      Folder root dataset — digunakan untuk menghitung rel_path.
        crop_dir_base:    Folder tempat cache crop wajah disimpan.
        prompt_groups:    List of (pos_prompts, neg_prompts) per label.
        thresholds:       List of float threshold per label.
        ambiguity_margin: Tidak digunakan, dipertahankan untuk kompatibilitas.
    """
    video_path:       str
    root_folder:      str
    crop_dir_base:    str
    prompt_groups:    List[Tuple[List[str], List[str]]]
    thresholds:       List[float]
    ambiguity_margin: float = 0.02


@app.post("/analyze_video")
def analyze_video(req: InferenceRequest):
    """
    Ekstrak frame dari video, crop wajah, dan jalankan inferensi SigLIP2.

    Flow:
        1. Validasi path video
        2. prepare_cropped_frames() -> ekstrak 6 frame + crop wajah
        3. run_siglip_on_frames()   -> inferensi label emosi
        4. Kembalikan hasil inferensi sebagai JSON
    """
    if not os.path.exists(req.video_path):
        raise HTTPException(status_code=400, detail="Path video tidak ditemukan")

    try:
        pil_images = prepare_cropped_frames(req.video_path, req.root_folder, req.crop_dir_base)
        if not pil_images:
            raise HTTPException(status_code=400, detail="Tidak dapat mengekstrak frame dari video")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal ekstrak frame: {e}")

    try:
        result = run_siglip_on_frames(
            pil_images, req.prompt_groups, req.thresholds, req.ambiguity_margin
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal inferensi SigLIP: {e}")


@app.get("/health")
def health_check():
    """Kembalikan status server. Digunakan untuk cek apakah service aktif."""
    return {"status": "ok"}
