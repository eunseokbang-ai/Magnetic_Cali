import numpy as np
import pandas as pd
import pytest

from mag_heading.batch import batch_compensate, compute_noise_metrics
from mag_heading.calibration import build_calibration
from mag_heading.heading import unit_vector
from mag_heading.io import read_pre_csv, write_pre_csv

BASELINE = 50000.0


def delta_b_true(theta, phi):
    return 2.0 * np.cos(theta) + 0.5 * np.sin(phi)


def _fibonacci_sphere(n, seed=0):
    i = np.arange(n)
    golden = (1 + 5**0.5) / 2
    theta = np.arccos(1 - 2 * (i + 0.5) / n)
    phi = (2 * np.pi * i / golden) % (2 * np.pi)
    return theta, phi


def _make_field_file(path, n=300, seed=0, swing_amplitude=1.0):
    """A synthetic field-acquisition file: real field is flat, but the sensor's
    heading oscillates sample-to-sample (like a swinging drone-suspended sensor),
    injecting exactly the heading-driven "swing" noise this whole tool targets.
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    theta = np.pi / 4 + swing_amplitude * 0.3 * np.sin(idx * 0.7)
    phi = 0.5 + swing_amplitude * 0.3 * np.cos(idx * 0.9)
    mag = BASELINE + delta_b_true(theta, phi) + rng.normal(scale=0.02, size=n)
    vec = unit_vector(theta, phi)
    df = pd.DataFrame(
        {"Mag": mag, "CompassX": vec[:, 0], "CompassY": vec[:, 1], "CompassZ": vec[:, 2]}
    )
    write_pre_csv(df, path)
    return df


def test_compute_noise_metrics_constant_signal_is_zero_noise():
    values = np.full(50, 123.0)
    metrics = compute_noise_metrics(values)
    assert metrics["std_nT"] == 0.0
    assert metrics["p2p_nT"] == 0.0
    assert metrics["motion_noise_std_nT"] == 0.0


def test_compute_noise_metrics_matches_manual_diff_std():
    rng = np.random.default_rng(0)
    values = rng.normal(size=200)
    metrics = compute_noise_metrics(values)
    assert metrics["motion_noise_std_nT"] == pytest.approx(np.std(np.diff(values)))
    assert metrics["std_nT"] == pytest.approx(np.std(values))
    assert metrics["p2p_nT"] == pytest.approx(values.max() - values.min())


def test_batch_compensate_reduces_motion_noise(tmp_path):
    theta_cal, phi_cal = _fibonacci_sphere(300)
    rng = np.random.default_rng(1)
    mag_cal = BASELINE + delta_b_true(theta_cal, phi_cal) + rng.normal(scale=0.02, size=300)
    vec = unit_vector(theta_cal, phi_cal)
    df_cal = pd.DataFrame(
        {"Mag": mag_cal, "CompassX": vec[:, 0], "CompassY": vec[:, 1], "CompassZ": vec[:, 2]}
    )
    model = build_calibration(df_cal, method="spline")

    paths = []
    for i in range(3):
        p = tmp_path / f"field{i}-pre.csv"
        _make_field_file(p, n=300, seed=10 + i)
        paths.append(str(p))

    results = batch_compensate(paths, model, keep_raw=True)

    assert len(results) == 3
    for r in results:
        assert r["comp_motion_noise_std_nT"] < r["raw_motion_noise_std_nT"]
        assert r["motion_noise_improvement_pct"] > 0

        reloaded = read_pre_csv(r["out_path"])
        assert "MagComp" in reloaded.columns
        reloaded_noise = compute_noise_metrics(reloaded["MagComp"].to_numpy())["motion_noise_std_nT"]
        assert reloaded_noise == pytest.approx(r["comp_motion_noise_std_nT"], rel=1e-6)


def test_batch_compensate_out_dir(tmp_path):
    theta_cal, phi_cal = _fibonacci_sphere(200)
    df_cal = pd.DataFrame(
        {
            "Mag": BASELINE + delta_b_true(theta_cal, phi_cal),
            "CompassX": unit_vector(theta_cal, phi_cal)[:, 0],
            "CompassY": unit_vector(theta_cal, phi_cal)[:, 1],
            "CompassZ": unit_vector(theta_cal, phi_cal)[:, 2],
        }
    )
    model = build_calibration(df_cal, method="nearest")

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    out_dir = tmp_path / "out"
    p = src_dir / "fieldA-pre.csv"
    _make_field_file(p, n=150, seed=20)

    results = batch_compensate([str(p)], model, out_dir=str(out_dir))

    assert len(results) == 1
    out_path = results[0]["out_path"]
    assert str(out_dir) in out_path
    assert (out_dir / "fieldA-pre-comp.csv").exists()


def test_batch_compensate_overwrite_mode(tmp_path):
    theta_cal, phi_cal = _fibonacci_sphere(200)
    df_cal = pd.DataFrame(
        {
            "Mag": BASELINE + delta_b_true(theta_cal, phi_cal),
            "CompassX": unit_vector(theta_cal, phi_cal)[:, 0],
            "CompassY": unit_vector(theta_cal, phi_cal)[:, 1],
            "CompassZ": unit_vector(theta_cal, phi_cal)[:, 2],
        }
    )
    model = build_calibration(df_cal, method="nearest")

    p = tmp_path / "field-pre.csv"
    _make_field_file(p, n=150, seed=30)

    results = batch_compensate([str(p)], model, keep_raw=False)
    reloaded = read_pre_csv(results[0]["out_path"])
    assert "MagComp" not in reloaded.columns
    assert "Mag" in reloaded.columns
