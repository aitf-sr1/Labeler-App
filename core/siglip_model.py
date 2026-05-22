"""
core/siglip_model.py

Singleton loader untuk model SigLIP2 (google/siglip2-base-patch16-224).
Model hanya dimuat sekali ke memori; pemanggilan berikutnya langsung return instance yang ada.
"""

import torch
from transformers import AutoProcessor, AutoModel

_siglip_model     = None
_siglip_processor = None
_device           = "cuda" if torch.cuda.is_available() else "cpu"
import threading

def get_device() -> str:
    return _device


_siglip_lock = threading.Lock()

def get_siglip():
    """Lazy-load SigLIP2. Return (model, processor) dalam eval mode."""
    global _siglip_model, _siglip_processor
    with _siglip_lock:
        if _siglip_model is None:
            import os
            MODEL_ID  = os.getenv("SIGLIP_MODEL_ID", "google/siglip2-base-patch16-224")
            use_fp16  = (_device == "cuda")
            dtype     = torch.float16 if use_fp16 else torch.float32
            dtype_str = "fp16" if use_fp16 else "fp32"
            print(f"Loading SigLIP2 on {_device.upper()} [{dtype_str}] ({MODEL_ID})...")
            _siglip_processor = AutoProcessor.from_pretrained(MODEL_ID)
            _siglip_model     = AutoModel.from_pretrained(MODEL_ID, torch_dtype=dtype)
            _siglip_model.to(_device)
            _siglip_model.eval()
            print("SigLIP2 ready.")
    return _siglip_model, _siglip_processor


def preload_siglip():
    """Panggil di background thread saat app start supaya model sudah siap sebelum dipakai."""
    threading.Thread(target=get_siglip, daemon=True, name="siglip-preload").start()
