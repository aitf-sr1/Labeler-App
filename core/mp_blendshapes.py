"""
core/mp_blendshapes.py
----------------------
Sumber blendshape ALTERNATIF: model `mp_blendshapes` dari Py-Feat (Hugging Face
`py-feat/mp_blendshapes`) — MLP-Mixer yang memetakan 146 titik MediaPipe Face Mesh
→ 52 blendshape ARKit (representasi otot FACS "secara longgar").

Ini OPSIONAL & STANDALONE: hanya butuh `torch` + `huggingface_hub` (untuk unduh
weights), TANPA install pustaka py-feat penuh. Kelas & konstanta di-VENDOR persis
dari source py-feat (cosanlab/py-feat, Apache-2.0) agar berjalan mandiri.

Pakai vs MediaPipe blendshape bawaan: di-toggle via env `BLENDSHAPE_SOURCE`
(lihat core/landmark_analyzer.py). Default tetap blendshape bawaan MediaPipe
(gratis, sudah ikut keluar dari FaceLandmarker). Modul ini untuk eksperimen A/B:
apakah mp_blendshapes lebih FACS-akurat (mis. AU4) daripada bawaan MediaPipe.

Arsitektur model di-port Google MediaPipe → PyTorch oleh L. Schoneveld
(deconstruct-mediapipe), dipublikasikan Py-Feat. Lihat DESIGN_RATIONALE §16.
"""

import threading
import torch
from torch import nn


# ── Nama 52 blendshape (urutan output model) — verbatim dari feat.utils ───────
MP_BLENDSHAPE_NAMES = [
    '_neutral', 'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft',
    'browOuterUpRight', 'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight',
    'eyeBlinkLeft', 'eyeBlinkRight', 'eyeLookDownLeft', 'eyeLookDownRight',
    'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft', 'eyeLookOutRight',
    'eyeLookUpLeft', 'eyeLookUpRight', 'eyeSquintLeft', 'eyeSquintRight',
    'eyeWideLeft', 'eyeWideRight', 'jawForward', 'jawLeft', 'jawOpen', 'jawRight',
    'mouthClose', 'mouthDimpleLeft', 'mouthDimpleRight', 'mouthFrownLeft',
    'mouthFrownRight', 'mouthFunnel', 'mouthLeft', 'mouthLowerDownLeft',
    'mouthLowerDownRight', 'mouthPressLeft', 'mouthPressRight', 'mouthPucker',
    'mouthRight', 'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower',
    'mouthShrugUpper', 'mouthSmileLeft', 'mouthSmileRight', 'mouthStretchLeft',
    'mouthStretchRight', 'mouthUpperUpLeft', 'mouthUpperUpRight', 'noseSneerLeft',
    'noseSneerRight',
]

# ── 146 indeks subset Face Mesh (dari 478) — verbatim dari feat.utils ─────────
MP_BLENDSHAPE_MODEL_LANDMARKS_SUBSET = [
    0, 1, 4, 5, 6, 7, 8, 10, 13, 14, 17, 21, 33, 37, 39, 40, 46, 52, 53, 54, 55,
    58, 61, 63, 65, 66, 67, 70, 78, 80, 81, 82, 84, 87, 88, 91, 93, 95, 103, 105,
    107, 109, 127, 132, 133, 136, 144, 145, 146, 148, 149, 150, 152, 153, 154, 155,
    157, 158, 159, 160, 161, 162, 163, 168, 172, 173, 176, 178, 181, 185, 191, 195,
    197, 234, 246, 249, 251, 263, 267, 269, 270, 276, 282, 283, 284, 285, 288, 291,
    293, 295, 296, 297, 300, 308, 310, 311, 312, 314, 317, 318, 321, 323, 324, 332,
    334, 336, 338, 356, 361, 362, 365, 373, 374, 375, 377, 378, 379, 380, 381, 382,
    384, 385, 386, 387, 388, 389, 390, 397, 398, 400, 402, 405, 409, 415, 454, 466,
    468, 469, 470, 471, 472, 473, 474, 475, 476, 477,
]


# ── Arsitektur MLP-Mixer (vendor persis dari py-feat, Apache-2.0) ─────────────
class MLPMixerLayer(nn.Module):
    def __init__(self, in_dim, num_patches, hidden_units_mlp1, hidden_units_mlp2,
                 dropout_rate=0.0,
                 eps1=0.0000010132789611816406, eps2=0.0000010132789611816406):
        super().__init__()
        self.mlp_token_mixing = nn.Sequential(
            nn.Conv2d(num_patches, hidden_units_mlp1, 1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(hidden_units_mlp1, num_patches, 1),
        )
        self.mlp_channel_mixing = nn.Sequential(
            nn.Conv2d(in_dim, hidden_units_mlp2, 1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(hidden_units_mlp2, in_dim, 1),
        )
        self.norm1 = nn.LayerNorm(in_dim, bias=False, elementwise_affine=True, eps=eps1)
        self.norm2 = nn.LayerNorm(in_dim, bias=False, elementwise_affine=True, eps=eps2)

    def forward(self, x):
        x_1 = self.norm1(x)
        mlp1_outputs = self.mlp_token_mixing(x_1)
        x = x + mlp1_outputs
        x_2 = self.norm2(x)
        mlp2_outputs = self.mlp_channel_mixing(x_2.permute(0, 3, 2, 1))
        x = x + mlp2_outputs.permute(0, 3, 2, 1)
        return x


class MediaPipeBlendshapesMLPMixer(nn.Module):
    def __init__(self, in_dim=64, num_patches=97, hidden_units_mlp1=384,
                 hidden_units_mlp2=256, num_blocks=4, dropout_rate=0.0, output_dim=52):
        super().__init__()
        self.conv1 = nn.Conv2d(146, 96, kernel_size=1)
        self.conv2 = nn.Conv2d(2, 64, kernel_size=1)
        self.extra_token = nn.Parameter(torch.randn(1, 64, 1, 1), requires_grad=True)
        self.mlpmixer_blocks = nn.Sequential(*[
            MLPMixerLayer(in_dim, num_patches, hidden_units_mlp1, hidden_units_mlp2, dropout_rate)
            for _ in range(num_blocks)
        ])
        self.output_mlp = nn.Conv2d(in_dim, output_dim, 1)

    def forward(self, x):
        x = x - x.mean(1, keepdim=True)
        x = x / x.norm(dim=2, keepdim=True).mean(1, keepdim=True)
        x = x.unsqueeze(-2) * 0.5
        x = self.conv1(x)
        x = x.permute(0, 3, 2, 1)
        x = self.conv2(x)
        extra_token_expanded = self.extra_token.expand(x.size(0), -1, -1, -1)
        x = torch.cat([extra_token_expanded, x], dim=3)
        x = x.permute(0, 3, 2, 1)
        x = self.mlpmixer_blocks(x)
        x = x.permute(0, 3, 2, 1)
        x = x[:, :, :, :1]
        x = self.output_mlp(x)
        x = torch.sigmoid(x)
        return x


# ── Loader & inferensi (lazy, thread-safe) ───────────────────────────────────
_lock = threading.Lock()
_model = None
_device = None


def _get_model():
    """Load model + weights sekali (lazy). Weights diunduh dari HF jika belum ada."""
    global _model, _device
    if _model is not None:
        return _model, _device
    with _lock:
        if _model is not None:
            return _model, _device
        from huggingface_hub import hf_hub_download
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        path = hf_hub_download(repo_id="py-feat/mp_blendshapes", filename="face_blendshapes.pth")
        ckpt = torch.load(path, map_location=_device)
        # weights bisa berupa state_dict langsung atau dibungkus {"net"/"state_dict": ...}
        if isinstance(ckpt, dict) and "net" in ckpt:
            ckpt = ckpt["net"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        m = MediaPipeBlendshapesMLPMixer()
        m.load_state_dict(ckpt)
        m.eval().to(_device)
        _model = m
        print(f"[mp_blendshapes] model siap — device={_device}")
        return _model, _device


def is_available() -> bool:
    """True jika model bisa dimuat (weights ter-download & state_dict cocok)."""
    try:
        _get_model()
        return True
    except Exception as e:
        print(f"[mp_blendshapes] tidak tersedia: {e}")
        return False


def compute_blendshapes(face_landmarks) -> dict:
    """
    Hitung 52 blendshape dari 478 MediaPipe Face Mesh landmark via model py-feat.

    Args:
        face_landmarks: list landmark MediaPipe (punya .x, .y; ternormalisasi 0–1),
                        panjang 478 (output FaceLandmarker).

    Returns:
        dict {nama_blendshape: nilai 0–1}, 52 entri (nama = MP_BLENDSHAPE_NAMES).
        Forward model scale/translation-invariant → koordinat normalized 0–1 cukup.
        Return {} jika landmark tidak lengkap atau model gagal.
    """
    if not face_landmarks or len(face_landmarks) < (max(MP_BLENDSHAPE_MODEL_LANDMARKS_SUBSET) + 1):
        return {}
    try:
        model, device = _get_model()
    except Exception:
        return {}
    pts = [[face_landmarks[i].x, face_landmarks[i].y] for i in MP_BLENDSHAPE_MODEL_LANDMARKS_SUBSET]
    x = torch.tensor(pts, dtype=torch.float32, device=device).unsqueeze(0)  # (1, 146, 2)
    with torch.no_grad():
        out = model(x).squeeze().detach().cpu().numpy()  # (52,)
    return {name: float(out[i]) for i, name in enumerate(MP_BLENDSHAPE_NAMES)}
