import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import tkinter as tk

from mag_heading.gui import MagHeadingApp
from mag_heading.heading import unit_vector
from mag_heading.io import write_pre_csv

BASELINE = 50000.0


def _write_survey(path, n=600, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    theta = np.pi / 4 + 0.3 * np.sin(idx * 0.05)
    phi = 0.5 + 0.3 * np.cos(idx * 0.07)
    mag = BASELINE + 2.0 * np.cos(theta) + rng.normal(scale=0.05, size=n)
    vec = unit_vector(theta, phi)
    df = pd.DataFrame(
        {"Mag": mag, "CompassX": vec[:, 0], "CompassY": vec[:, 1], "CompassZ": vec[:, 2]}
    )
    write_pre_csv(df, path)


def _write_multi_leg_survey(path, leg_len=1000, seed=0):
    # Four well-separated, internally clean headings (like four boustrophedon
    # legs) -- no single short window sees all four, so a passing auto-build
    # result here demonstrates genuine multi-segment combination, not a fluke.
    rng = np.random.default_rng(seed)
    centers = [(0.7, 0.3), (0.7, 2.0), (0.7, -2.0), (0.7, -1.0)]
    thetas, phis = [], []
    for theta_c, phi_c in centers:
        thetas.append(np.full(leg_len, theta_c) + rng.normal(scale=0.03, size=leg_len))
        phis.append(np.full(leg_len, phi_c) + rng.normal(scale=0.03, size=leg_len))
    theta = np.concatenate(thetas)
    phi = np.concatenate(phis)
    mag = BASELINE + 2.0 * np.cos(theta) + 0.5 * np.sin(phi) + rng.normal(scale=0.05, size=len(theta))
    vec = unit_vector(theta, phi)
    df = pd.DataFrame(
        {"Mag": mag, "CompassX": vec[:, 0], "CompassY": vec[:, 1], "CompassZ": vec[:, 2]}
    )
    write_pre_csv(df, path)


@pytest.fixture(scope="module")
def app():
    # module-scoped: creating a second real Tk() root right after destroying the
    # first one is flaky (Tcl interpreter teardown/setup race -- "invalid command
    # name tcl_findLibrary"), so tests in this file share one root instead.
    # settings_path=None: never read/write the user's real ~/.mag_heading_gui.json;
    # sync_tasks=True: scan/auto-build/compensate run inline so asserts see results.
    application = MagHeadingApp(settings_path=None, sync_tasks=True)
    yield application
    application.destroy()


def test_loading_new_file_clears_stale_scan_candidates(tmp_path, app):
    path_a = tmp_path / "surveyA-pre.csv"
    path_b = tmp_path / "surveyB-pre.csv"
    _write_survey(path_a, n=600, seed=1)
    _write_survey(path_b, n=400, seed=2)

    app._load_pre(str(path_a))
    app.window_size_var.set(200)
    app._on_scan_clicked()
    assert len(app._candidates) > 0

    app.start_idx, app.end_idx = app._candidates[0]["start"], app._candidates[0]["end"]
    app._on_build_clicked()
    assert app.model is not None

    app._load_pre(str(path_b))

    assert app._candidates == []
    assert app.candidates_list.size() == 0
    # the calibration itself is file-independent and is kept across loads
    # (load calibration -> load each survey file -> compensate is the main workflow)
    assert app.model is not None
    assert app.start_idx is None
    assert app.end_idx is None
    assert app.selection_label.cget("text") == "start=- end=- (drag on plot to select)"


def test_build_from_selection_uses_currently_loaded_file(tmp_path, app):
    path_a = tmp_path / "surveyA-pre.csv"
    path_b = tmp_path / "surveyB-pre.csv"
    _write_survey(path_a, n=600, seed=3)
    _write_survey(path_b, n=600, seed=4)

    app._load_pre(str(path_a))
    app.window_size_var.set(200)
    app._on_scan_clicked()
    candidate = app._candidates[0]

    # Switch files, then reuse the same index range the stale candidate pointed at --
    # must build against file B's data, not silently reuse anything from file A.
    app._load_pre(str(path_b))
    app.start_idx, app.end_idx = candidate["start"], candidate["end"]
    app._on_build_clicked()

    assert app.model is not None
    expected_mag = app.df["Mag"].to_numpy()[app.start_idx : app.end_idx + 1]
    np.testing.assert_allclose(app.model.mag_cal, expected_mag)


def test_add_segment_accumulates_and_clears_pending_selection(tmp_path, app):
    path = tmp_path / "surveyC-pre.csv"
    _write_survey(path, n=600, seed=5)
    app._load_pre(str(path))

    app.start_idx, app.end_idx = 10, 20
    app._on_add_segment_clicked()

    assert app.segments == [(10, 20)]
    assert app.segments_list.size() == 1
    # pending selection is cleared so the next drag starts fresh
    assert app.start_idx is None
    assert app.end_idx is None


def test_build_calibration_uses_all_added_segments(tmp_path, app):
    path = tmp_path / "surveyD-pre.csv"
    _write_survey(path, n=600, seed=6)
    app._load_pre(str(path))

    app.start_idx, app.end_idx = 10, 59
    app._on_add_segment_clicked()
    app.start_idx, app.end_idx = 200, 249
    app._on_add_segment_clicked()
    assert len(app.segments) == 2

    app.center_segments_var.set(False)
    app._on_build_clicked()

    assert app.model is not None
    mag = app.df["Mag"].to_numpy()
    expected = np.concatenate([mag[10:60], mag[200:250]])
    np.testing.assert_allclose(app.model.mag_cal, expected)


def test_remove_and_clear_segments(tmp_path, app):
    path = tmp_path / "surveyE-pre.csv"
    _write_survey(path, n=600, seed=7)
    app._load_pre(str(path))

    app.start_idx, app.end_idx = 10, 20
    app._on_add_segment_clicked()
    app.start_idx, app.end_idx = 100, 110
    app._on_add_segment_clicked()
    assert len(app.segments) == 2

    app.segments_list.selection_set(0)
    app._on_remove_segment_clicked()
    assert app.segments == [(100, 110)]
    assert app.segments_list.size() == 1

    app._on_clear_segments_clicked()
    assert app.segments == []
    assert app.segments_list.size() == 0


def test_loading_new_file_clears_accumulated_segments(tmp_path, app):
    path_a = tmp_path / "surveyF-pre.csv"
    path_b = tmp_path / "surveyG-pre.csv"
    _write_survey(path_a, n=600, seed=8)
    _write_survey(path_b, n=600, seed=9)

    app._load_pre(str(path_a))
    app.start_idx, app.end_idx = 10, 20
    app._on_add_segment_clicked()
    assert len(app.segments) == 1

    app._load_pre(str(path_b))
    assert app.segments == []
    assert app.segments_list.size() == 0


def test_auto_build_populates_segments_and_saves_on_pass(tmp_path, app):
    path = tmp_path / "surveyAuto-pre.csv"
    _write_multi_leg_survey(path, leg_len=1000, seed=50)
    app._load_pre(str(path))

    app.auto_window_sizes_var.set("400")
    app.auto_coverage_target_var.set(0.05)
    app.auto_max_segments_var.set(30)
    app.max_var_var.set(1.0)

    save_path = tmp_path / "auto_result.npz"
    with patch("mag_heading.gui.filedialog.asksaveasfilename", return_value=str(save_path)):
        app._on_auto_build_clicked()

    assert app.model is not None
    assert len(app.segments) >= 1
    assert app.segments_list.size() == len(app.segments)
    # a passing result must be saved automatically (via the normal save dialog),
    # not left for the user to click Save separately
    assert save_path.exists()


def test_auto_build_does_not_save_when_not_passed(tmp_path, app):
    # An impossible-to-reach coverage target guarantees the stopping criteria
    # are never met even though individual segments accept fine -- the save
    # dialog must never be triggered for a non-passing best-effort result.
    path = tmp_path / "surveyAutoFail-pre.csv"
    _write_survey(path, n=800, seed=60)
    app._load_pre(str(path))

    app.auto_window_sizes_var.set("200")
    app.auto_coverage_target_var.set(-1.0)
    app.auto_max_segments_var.set(5)
    app.max_var_var.set(1.0)

    with patch("mag_heading.gui.filedialog.asksaveasfilename") as mock_save_dialog, \
            patch("mag_heading.gui.messagebox.showwarning") as mock_warning:
        app._on_auto_build_clicked()

    mock_save_dialog.assert_not_called()
    mock_warning.assert_called_once()
    assert app.model is not None


def _write_gps_survey(path, n=1200, seed=0, with_time=True):
    """Two straight N/S lines joined by a turn, 10 Hz, with heading-dependent Mag."""
    rng = np.random.default_rng(seed)
    third = n // 3
    lat0, lon0 = 34.5, 126.5
    north = np.linspace(0, 300, third)
    turn_t = np.linspace(0, np.pi, n - 2 * third)
    lat = np.concatenate([lat0 + north / 111000.0,
                          lat0 + (300 + 20 * np.sin(turn_t)) / 111000.0,
                          lat0 + north[::-1] / 111000.0])
    lon = np.concatenate([np.full(third, lon0),
                          lon0 + (20 - 20 * np.cos(turn_t)) / 91000.0,
                          np.full(third, lon0 + 40 / 91000.0)])
    phi = np.concatenate([np.full(third, 0.3), 0.3 + turn_t, np.full(third, 0.3 + np.pi)])
    phi = np.angle(np.exp(1j * phi)) + rng.normal(scale=0.02, size=n)
    theta = np.full(n, 0.9) + 0.2 * np.sin(np.arange(n) * 0.05)
    mag = BASELINE + 2.0 * np.cos(theta) + 0.5 * np.sin(phi) + rng.normal(scale=0.05, size=n)
    vec = unit_vector(theta, phi)
    data = {"Mag": mag, "CompassX": vec[:, 0], "CompassY": vec[:, 1], "CompassZ": vec[:, 2],
            "Latitude": lat, "Longitude": lon}
    if with_time:
        t = pd.to_timedelta(np.arange(n) * 0.1, unit="s")
        data = {"Time": [str(x).split()[-1] for x in t], **data}
    write_pre_csv(pd.DataFrame(data), path)


def test_preview_compensation_reports_turn_and_line_noise(tmp_path, app):
    path = tmp_path / "gps-pre.csv"
    _write_gps_survey(path, seed=11)
    app._load_pre(str(path))
    app.start_idx, app.end_idx = 0, len(app.df) - 1
    app._on_build_clicked()
    app.smooth_enabled_var.set(True)
    app.smooth_hz_var.set(1.0)

    app._on_preview_clicked()

    assert app.compensated is not None
    assert "MagComp" in app.compensated.columns
    summary = app.result_summary.cget("text")
    assert "turns" in summary and "lines" in summary
    assert "smoothing 1 Hz" in summary
    assert len(app._result_frames) == 2  # time series + 2D map
    assert not app._busy


def test_compensate_without_time_column_falls_back_to_no_smoothing(tmp_path, app):
    path = tmp_path / "gps-notime-pre.csv"
    _write_gps_survey(path, seed=12, with_time=False)
    app._load_pre(str(path))
    app.start_idx, app.end_idx = 0, len(app.df) - 1
    app._on_build_clicked()
    app.smooth_enabled_var.set(True)

    out = tmp_path / "gps-notime-comp.csv"
    with patch("mag_heading.gui.filedialog.asksaveasfilename", return_value=str(out)):
        app._on_compensate_clicked()

    assert out.exists()
    assert "no smoothing" in app.result_summary.cget("text")


def test_batch_keeps_going_past_a_bad_file(tmp_path, app):
    good = tmp_path / "good-pre.csv"
    _write_gps_survey(good, seed=13)
    bad = tmp_path / "bad-pre.csv"
    bad.write_text('not,a,magarrow' + chr(10) + 'file,at,all' + chr(10))
    app._load_pre(str(good))
    app.start_idx, app.end_idx = 0, len(app.df) - 1
    app._on_build_clicked()

    app._on_batch_clear()
    app.batch_files_list.insert(tk.END, str(bad))
    app.batch_files_list.insert(tk.END, str(good))
    app.batch_out_dir_var.set(str(tmp_path / "out"))
    app._on_batch_run_clicked()

    rows = [app.batch_tree.item(r) for r in app.batch_tree.get_children()]
    assert len(rows) == 2
    assert "error" in rows[0]["tags"]
    assert rows[1]["values"][-1].endswith("-comp.csv")
    assert "FAILED" in app.batch_summary_label.cget("text")
    assert (tmp_path / "out" / "good-pre-comp.csv").exists() or list((tmp_path / "out").glob("*.csv"))


def test_report_shows_pass_fail_header(tmp_path, app):
    path = tmp_path / "surveyR-pre.csv"
    _write_survey(path, n=600, seed=21)
    app._load_pre(str(path))
    app.start_idx, app.end_idx = 0, 599
    app.max_var_var.set(1000.0)
    app._on_build_clicked()
    first_line = app.report_text.get("1.0", "1.end")
    assert first_line in ("PASS", "FAIL")
    assert "self-comp p2p" in app.report_text.get("1.0", tk.END)


def test_background_task_runs_off_the_ui_thread(tmp_path, app):
    import threading
    import time

    seen = {}
    app.sync_tasks = False
    try:
        def work(report):
            seen["thread"] = threading.current_thread()
            report("halfway")
            return 42

        app._run_task("demo", work, lambda r: seen.setdefault("result", r))
        assert app._run_task("second", work, lambda r: None) is False  # busy: rejected
        deadline = time.time() + 5
        while "result" not in seen and time.time() < deadline:
            app.update()
            time.sleep(0.02)
    finally:
        app.sync_tasks = True
    assert seen["result"] == 42
    assert seen["thread"] is not threading.main_thread()
    assert app._busy is None


def test_find_pre_files_accepts_lpf_names_and_skips_outputs(tmp_path):
    from mag_heading.gui import find_pre_files

    for name in ["A-10Hz-pre-LPF.csv", "B-pre.csv", "B-pre-comp.csv", "C-pre-LPF-comp.csv",
                 "D-pre-leveled.csv", "raw.csv", "notes.txt"]:
        (tmp_path / name).write_text("x")
    names = [os.path.basename(p) for p in find_pre_files(str(tmp_path))]
    assert names == ["A-10Hz-pre-LPF.csv", "B-pre.csv"]
