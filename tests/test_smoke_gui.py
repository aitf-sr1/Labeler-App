"""
Uji asap (smoke test) GUI — bangun seluruh aplikasi tanpa interaksi user untuk
menangkap error konstruksi/regresi (mis. salah nama argumen, widget hilang).

Cara pakai (butuh display; di server pakai xvfb-run / DISPLAY virtual):
    cd Labeler-App-Siglip-2
    ../../.venv/bin/python tests/test_smoke_gui.py

Juga kompatibel pytest:  pytest tests/test_smoke_gui.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)


def main():
    import numpy as np
    import customtkinter as ctk
    import app as mod

    root = ctk.CTk()
    try:
        a = mod.VideoLabelerApp(root)

        # Panel LP: build lazy + semua getter publik
        a.left_panel.show_lp_mode()
        lp = a.left_panel.lp_panel_content
        assert lp._built, "panel LP gagal build"
        lp.get_selected_emotions(); lp.get_target_n(); lp.get_picked_indices()
        lp.get_driving_folder(); lp.get_driving_list("Confusion"); lp.get_merge_mode()
        lp.get_face_folder(); lp.get_selected_faces(); lp.get_picked_fractions()

        # Kunci posisi proporsional: tandai di video pendek lalu ganti ke video panjang
        import numpy as _np
        vid_a = [_np.zeros((40, 40, 3), "uint8")] * 10   # hasil driving A (10 frame)
        vid_b = [_np.zeros((40, 40, 3), "uint8")] * 50   # hasil driving B (50 frame)
        lp.set_result_frames(vid_a, "Confusion")
        lp.frame_terpilih = {2, 8}; lp._simpan_fraksi()   # tandai 2 frame (frac 0.22, 0.89)
        lp.set_result_frames(vid_b, "Confusion")          # ganti driving → remap proporsional
        assert len(lp.get_picked_indices()) == 2, "jumlah frame tertanda harus tetap 2"
        assert max(lp.get_picked_indices()) > 9, "posisi harus menyesuaikan video baru (50 frame)"
        # _lp_extract_indices: jumlah sama untuk total beda
        assert len(a._lp_extract_indices([0.0, 0.5, 1.0], 4, 80)) == 3
        assert a._lp_extract_indices([0.0, 0.5, 1.0], 4, 80) == [0, 40, 79]

        # Status & loading
        lp.update_progress("uji"); lp.start_loading("uji loading"); lp.stop_loading("selesai")
        lp.clear_source("uji"); lp.set_source_label("abcd1234", 0); lp.reset()

        # Tinjau & navigasi (ribuan item): set_review_data + prev/next/loncat/terapkan_state
        dummy = np.zeros((80, 80, 3), dtype="uint8")
        label_kosong = {l: 0 for l in mod.LABELS}
        items = [(f"/tmp/x{i}.jpg", "Confusion", f"rel/x{i}.jpg") for i in range(120)]
        labels = {f"rel/x{i}.jpg": dict(label_kosong, Confusion=1) for i in range(120)}
        lp.set_review_data(items, labels, {"rel/x0.jpg": {"Confusion": 1}}, set())
        assert lp.idx_tinjau == 0
        lp._tinjau_geser(+1); assert lp.idx_tinjau == 1
        lp.entri_loncat.delete(0, "end"); lp.entri_loncat.insert(0, "95"); lp._tinjau_loncat()
        assert lp.idx_tinjau == 94, lp.idx_tinjau
        lp.terapkan_state("rel/x94.jpg", ditolak=True)
        assert "rel/x94.jpg" in lp.review_ditolak
        lp._geser_halaman(+1)                  # ganti halaman grid
        lp.set_review_data([], {}, {}, set())  # kasus kosong
        lp.render_faces([("/tmp/a.jpg", False, dummy), ("/tmp/b.jpg", True, dummy)])

        # Metode app yang dipanggil panel harus ada
        for m in ["_goto_lp_mark", "_lp_cancel", "_lp_clear_all_marks", "_lp_delete_rejected",
                  "_lp_label_all_ai", "_lp_merge_into_label2d", "_lp_undo_merge",
                  "_lp_process_current", "_lp_process_batch", "_lp_process_faces",
                  "_lp_refresh_review", "_lp_refresh_faces", "_lp_save_frames",
                  "_lp_scan_driving", "_lp_set_label", "_lp_toggle_reject",
                  "_lp_build_merged_dataset", "_rel_to_idx"]:
            assert hasattr(a, m), f"metode app hilang: {m}"

        # Ganti-ganti mode tidak boleh error
        a.left_panel.show_gallery_mode()
        a.left_panel.show_rules_mode()
        a.left_panel.show_gallery_mode()

        root.update()
        print("SMOKE TEST: ALL OK")
    finally:
        root.destroy()


def test_smoke_gui():
    """Entry pytest."""
    main()


if __name__ == "__main__":
    main()
