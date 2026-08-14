import numpy as np
import pandas as pd
import pytest

from mag_heading.calibration import build_calibration
from mag_heading.compensation import apply_compensation, write_comp_csv
from mag_heading.heading import compute_heading, unit_vector
from mag_heading.io import read_pre_csv, comp_output_path

BASELINE = 50000.0


def delta_b_true(theta, phi):
    return 2.0 * np.cos(theta) + 0.5 * np.sin(phi)


def _fibonacci_sphere(n, seed=0):
    i = np.arange(n)
    golden = (1 + 5**0.5) / 2
    theta = np.arccos(1 - 2 * (i + 0.5) / n)
    phi = (2 * np.pi * i / golden) % (2 * np.pi)
    return theta, phi


def _make_df(theta, phi, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    mag = BASELINE + delta_b_true(theta, phi) + rng.normal(scale=noise, size=len(theta))
    vec = unit_vector(theta, phi)
    return pd.DataFrame(
        {
            "Mag": mag,
            "CompassX": vec[:, 0],
            "CompassY": vec[:, 1],
            "CompassZ": vec[:, 2],
        }
    )


def test_apply_compensation_keep_raw_adds_column():
    theta, phi = _fibonacci_sphere(100)
    df = _make_df(theta, phi)
    model = build_calibration(df, method="spline")

    out = apply_compensation(df, model, keep_raw=True)
    assert "MagComp" in out.columns
    assert "Mag" in out.columns
    np.testing.assert_allclose(out["Mag"], df["Mag"])

    # apply_compensation must match the same theta/phi -> predict -> subtract
    # pipeline it wraps, independent of how well the model fits the true
    # heading-error function.
    theta_r, phi_r = compute_heading(df["CompassX"], df["CompassY"], df["CompassZ"])
    expected = df["Mag"].to_numpy() - model.predict(theta_r, phi_r)
    np.testing.assert_allclose(out["MagComp"], expected, atol=1e-8)


def test_apply_compensation_overwrite_replaces_mag():
    theta, phi = _fibonacci_sphere(100)
    df = _make_df(theta, phi)
    model = build_calibration(df, method="spline")

    out = apply_compensation(df, model, keep_raw=False)
    assert "MagComp" not in out.columns

    theta_r, phi_r = compute_heading(df["CompassX"], df["CompassY"], df["CompassZ"])
    expected = df["Mag"].to_numpy() - model.predict(theta_r, phi_r)
    np.testing.assert_allclose(out["Mag"], expected, atol=1e-8)


def test_comp_output_path_naming():
    assert comp_output_path("survey-pre.csv") == "survey-pre-comp.csv"
    assert comp_output_path("survey-pre") == "survey-pre-comp.csv"


def test_write_comp_csv_round_trip(tmp_path):
    theta, phi = _fibonacci_sphere(50)
    df = _make_df(theta, phi)
    model = build_calibration(df, method="nearest")
    compensated = apply_compensation(df, model, keep_raw=True)

    original_path = tmp_path / "survey-pre.csv"
    out_path = write_comp_csv(compensated, original_path)

    assert str(out_path) == str(tmp_path / "survey-pre-comp.csv")
    reloaded = read_pre_csv(out_path)
    np.testing.assert_allclose(reloaded["MagComp"], compensated["MagComp"], atol=1e-6)
