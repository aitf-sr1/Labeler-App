from .rules import load_rules, save_rules, DEFAULT_RULES

# Heavy submodules (inference, siglip_model, face_detector, recalculate)
# are NOT imported here to keep app startup fast.
# Import them directly from their modules when needed:
#   from core.inference import run_siglip_on_frames, run_siglip_batch
#   from core.recalculate import recalculate_batch
#   from core.siglip_model import preload_siglip, get_siglip
#   from core.face_detector import crop_face
