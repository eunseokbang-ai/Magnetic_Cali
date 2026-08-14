from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

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
    application = MagHeadingApp()
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
    assert app.model is None
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
