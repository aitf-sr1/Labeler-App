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

        # Tanda PER VIDEO DRIVING: ganti driving -> tanda video itu dipulihkan persis
        lp.driving_aktif = "/d/Confusion1.mp4"
        lp.tanda_per_driving["/d/Confusion1.mp4"] = [2, 8]
        lp.set_result_frames(vid_b, "Confusion")
        assert lp.get_picked_indices() == [2, 8], lp.get_picked_indices()
        lp.driving_aktif = ""; lp.tanda_per_driving.clear()
        # Scrub driving SEBELUM proses: tanpa hasil, slider pakai panjang driving
        lp.frame_hasil = []
        lp.set_driving_frames(vid_a, "Confusion", path="/d/Confusion1.mp4")
        assert lp._total_aktif() == 10 and lp.driving_aktif.endswith("Confusion1.mp4")
        lp._tandai_frame_sekarang()
        assert lp.tanda_per_driving["/d/Confusion1.mp4"], "tanda harus tersimpan per driving"
        lp.driving_aktif = ""; lp.tanda_per_driving.clear(); lp._bersihkan_tanda()
        # Dropdown tanpa 'Semua': maks 1 video per emosi; '(tidak ada)' -> kosong
        assert all(len(lp.get_driving_list(e)) <= 1 for e in mod.LABELS)
        lp.pilihan_driving["Confusion"].set("(tidak ada)")
        assert lp.get_driving_list("Confusion") == []
        # Emosi single-select: pilih kedua -> hanya yang terakhir aktif; klik lagi -> mati
        lp._ganti_emosi("Confusion"); lp._ganti_emosi("Frustration")
        assert lp.get_selected_emotions() == ["Frustration"], lp.get_selected_emotions()
        lp._ganti_emosi("Frustration")
        assert lp.get_selected_emotions() == []

        # Status & loading
        lp.update_progress("uji"); lp.start_loading("uji loading"); lp.stop_loading("selesai")
        lp.clear_source("uji"); lp.set_source_label("abcd1234", 0); lp.reset()

        # Lokasi simpan + throttle refresh (anti-lag) tidak boleh error
        lp.set_save_info("uji lokasi")
        a._lp_update_save_info()
        a._lp_last_review_ts = 0.0
        a._lp_refresh_review_throttled(jeda=999)   # sekali jalan
        a._lp_refresh_review_throttled(jeda=999)   # ke-2 harus di-skip (throttle)

        # Tinjau & navigasi (ribuan item): set_review_data + prev/next/loncat/terapkan_state
        dummy = np.zeros((80, 80, 3), dtype="uint8")
        label_kosong = {l: 0 for l in mod.LABELS}
        items = [(f"/tmp/x{i}.jpg", "Confusion", f"rel/x{i}.jpg") for i in range(120)]
        labels = {f"rel/x{i}.jpg": dict(label_kosong, Confusion=1) for i in range(120)}
        lp.set_review_data(items, labels, {"rel/x0.jpg": {"Confusion": 1}}, set())
        assert lp.idx_tinjau == 0
        lp._tinjau_geser(+1); assert lp.idx_tinjau == 1
        lp.entri_loncat.focus_set()   # simulasi kursor di kotak Loncat
        lp.entri_loncat.delete(0, "end"); lp.entri_loncat.insert(0, "95"); lp._tinjau_loncat()
        assert lp.idx_tinjau == 94, lp.idx_tinjau
        # Regresi: setelah Loncat, fokus dilepas dari Entry → panah HARUS tetap menavigasi
        root.update()
        ent = getattr(lp.entri_loncat, "_entry", lp.entri_loncat)
        assert root.focus_get() is not ent, "fokus masih nyangkut di kotak Loncat"
        a._on_arrow(+1); assert lp.idx_tinjau == 95, ("panah mati setelah Loncat", lp.idx_tinjau)
        lp.terapkan_state("rel/x94.jpg", ditolak=True)
        assert "rel/x94.jpg" in lp.review_ditolak
        lp._geser_halaman(+1)                  # ganti halaman grid
        # Filter tinjau: Ditolak -> 1 item; AI != target -> item dgn AI tapi target=0
        lp.var_filter_tinjau.set("Ditolak"); lp._terapkan_filter()
        assert len(lp.review_items) == 1, len(lp.review_items)
        lp.review_ai["rel/x1.jpg"] = {l: 0 for l in mod.LABELS}    # AI tak deteksi target
        lp.var_filter_tinjau.set("AI != target"); lp._terapkan_filter()
        assert len(lp.review_items) == 1 and lp.review_items[0][2] == "rel/x1.jpg"
        lp.var_filter_tinjau.set("Semua"); lp._terapkan_filter()
        assert len(lp.review_items) == 120
        # Filter 'Dibuang (_trash)': gambar yang dibuang TETAP terlihat & bisa dipulihkan
        buang = [(f"/tmp/_trash/t{i}.jpg", "Confusion", f"rel/t{i}.jpg") for i in range(3)]
        lp.set_review_data(items, labels, {}, set(), dibuang=buang)
        lp.var_filter_tinjau.set("Dibuang (_trash)"); lp._terapkan_filter()
        assert len(lp.review_items) == 3, len(lp.review_items)
        lp.var_filter_tinjau.set("Semua"); lp._terapkan_filter()
        assert len(lp.review_items) == 120

        # Panah keyboard sadar-mode: di panel LP panah menggerakkan pemeriksa hasil
        root.update()
        assert a._arrow_target_lp() is lp, "panel LP tampil → panah harus ke pemeriksa"
        lp.idx_tinjau = 0; lp._render_pemeriksa()
        a._on_arrow(+1); assert lp.idx_tinjau == 1, lp.idx_tinjau
        a._on_arrow(-1); assert lp.idx_tinjau == 0, lp.idx_tinjau

        # Seksi panel bisa dilipat: toggle 2x tidak error & state kembali
        assert lp._seksi_toggle, "seksi lipat belum terdaftar"
        for _nama, toggle in lp._seksi_toggle.items():
            toggle(); toggle()

        # Cache hasil persisten: kunci = identitas file (ukuran+mtime), BUKAN nama —
        # file driving di-rename tetap dikenali; emosi lain = kunci lain
        import tempfile, shutil as _sh
        tdir = tempfile.mkdtemp(prefix="lp_pcache_")
        src = os.path.join(tdir, "src.jpg")
        drv = os.path.join(tdir, "drv.mp4")
        out = os.path.join(tdir, "out.mp4")
        for p, isi in [(src, b"a" * 100), (drv, b"b" * 200), (out, b"c" * 300)]:
            with open(p, "wb") as f:
                f.write(isi)
        path_lama = getattr(a, "path_json_augment", None)
        a.path_json_augment = os.path.join(tdir, "augment_marks.json")
        assert a._lp_sig(src), "sig file harus terisi"
        a._lp_pcache_put(src, drv, "Confusion", out)
        assert a._lp_pcache_get(src, drv, "Confusion") == out
        drv_baru = os.path.join(tdir, "NamaBaru.mp4")
        os.rename(drv, drv_baru)                      # rename = video yang sama
        assert a._lp_pcache_get(src, drv_baru, "Confusion") == out
        assert a._lp_pcache_get(src, drv_baru, "Frustration") is None
        # Reverse-lookup: render 'tetep ada' saat frame sumber dibuka lagi (lepas emosi)
        tdir2 = tempfile.mkdtemp(prefix="lp_any_")
        a.path_json_augment = os.path.join(tdir2, "augment_marks.json")
        root_lama, vf_lama = a.root_folder, a.video_files
        a.root_folder = tdir2
        vidX = os.path.join(tdir2, "vidX.mp4")
        with open(vidX, "wb") as f:
            f.write(b"v" * 10)
        a.video_files = [vidX]
        srcdir = os.path.join(tdir2, "cropped_faces", "clean", "vidX")
        os.makedirs(srcdir, exist_ok=True)
        srcf = os.path.join(srcdir, "frame_00.jpg")
        outv = os.path.join(tdir2, "render.mp4")
        drvv = os.path.join(tdir2, "drv2.mp4")
        for p, isi in [(srcf, b"s" * 50), (outv, b"o" * 60), (drvv, b"d" * 70)]:
            with open(p, "wb") as f:
                f.write(isi)
        a._lp_pcache_put(srcf, drvv, "Boredom", outv)
        got = a._lp_pcache_lookup_any_for_source((0, 0))
        assert got and got["video"] == outv and got["emo"] == "Boredom", got
        a.root_folder, a.video_files = root_lama, vf_lama
        a.path_json_augment = path_lama
        _sh.rmtree(tdir2, ignore_errors=True)

        # Panah saat TIDAK ada daftar tinjau → scrub video HASIL yang sedang tampil
        lp.set_review_data([], {}, {}, set())
        lp.set_result_frames([np.zeros((30, 30, 3), "uint8")] * 8, "Confusion")
        lp.index_hasil = 0
        assert lp.geser_frame_relatif(+1) is True and lp.index_hasil == 1
        root.update()
        a._on_arrow(+1); assert lp.index_hasil == 2, lp.index_hasil
        lp.set_result_frames([], "Confusion"); lp.frame_driving = []
        assert lp.geser_frame_relatif(+1) is False

        lp.set_review_data([], {}, {}, set())  # kasus kosong
        lp.render_faces([("/tmp/a.jpg", False, dummy), ("/tmp/b.jpg", True, dummy)])

        # Metode app yang dipanggil panel harus ada
        for m in ["_goto_lp_mark", "_lp_cancel", "_lp_clear_all_marks", "_lp_delete_rejected",
                  "_lp_label_all_ai", "_lp_merge_into_label2d", "_lp_undo_merge",
                  "_lp_process_current", "_lp_process_batch", "_lp_process_faces",
                  "_lp_refresh_review", "_lp_refresh_faces", "_lp_save_frames",
                  "_lp_scan_driving", "_lp_set_label", "_lp_toggle_reject",
                  "_lp_build_merged_dataset", "_rel_to_idx",
                  "_lp_show_stats", "_lp_auto_reject_mismatch",
                  "_lp_restore_trash", "_lp_restore_one", "_lp_list_trashed",
                  "_lp_pcache_get", "_lp_pcache_put", "_lp_pcache_lookup_any_for_source",
                  "_on_arrow", "_lp_update_save_info", "_lp_refresh_review_throttled"]:
            assert hasattr(a, m), f"metode app hilang: {m}"
        for m in ["_lepas_fokus", "_tinjau_geser", "_tinjau_loncat", "_buka_di_pemeriksa",
                  "_geser_halaman"]:
            assert hasattr(lp, m), f"metode panel hilang: {m}"

        # Ganti-ganti mode tidak boleh error; di galeri panah kembali ke navigasi video
        a.left_panel.show_gallery_mode()
        root.update()
        assert a._arrow_target_lp() is None, "galeri → panah harus ke navigasi video"
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
