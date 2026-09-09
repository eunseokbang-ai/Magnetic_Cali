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


def _survey_with_compass_noise(n=2000, rate_hz=10.0, seed=0):
    """A straight-line survey: heading nearly fixed, compass carries broadband noise.

    This is the case where the correction hurts -- the true heading effect is
    almost constant, so anything the correction adds sample-to-sample is noise.
    """
    rng = np.random.default_rng(seed)
    theta = np.full(n, 0.9) + 0.02 * np.sin(np.arange(n) * 0.01)
    phi = np.full(n, 0.4) + 0.02 * np.cos(np.arange(n) * 0.008)
    vec = unit_vector(theta, phi)
    vec = vec + rng.normal(scale=0.01, size=vec.shape)  # compass noise
    mag = BASELINE + delta_b_true(theta, phi) + rng.normal(scale=0.02, size=n)
    t = pd.to_timedelta(np.arange(n) / rate_hz, unit="s")
    return pd.DataFrame(
        {
            "Time": [str(x)[-15:] for x in t],
            "Mag": mag,
            "CompassX": vec[:, 0],
            "CompassY": vec[:, 1],
            "CompassZ": vec[:, 2],
        }
    )


def test_smoothing_the_correction_reduces_injected_noise():
    theta_c, phi_c = _fibonacci_sphere(600, seed=11)
    model = build_calibration(_make_df(theta_c, phi_c), method="nearest")

    survey = _survey_with_compass_noise(seed=12)
    raw_jitter = np.std(np.diff(survey["Mag"].to_numpy()))

    unsmoothed = apply_compensation(survey, model)["MagComp"].to_numpy()
    smoothed = apply_compensation(survey, model, smooth_hz=1.0)["MagComp"].to_numpy()

    # without smoothing the correction adds sample-to-sample noise; with it, less
    assert np.std(np.diff(unsmoothed)) > raw_jitter
    assert np.std(np.diff(smoothed)) < np.std(np.diff(unsmoothed))


def test_smooth_hz_infers_sample_rate_from_time_column():
    theta_c, phi_c = _fibonacci_sphere(400, seed=13)
    model = build_calibration(_make_df(theta_c, phi_c), method="nearest")
    survey = _survey_with_compass_noise(n=500, rate_hz=10.0, seed=14)

    from mag_heading.compensation import infer_sample_rate_hz

    assert infer_sample_rate_hz(survey) == pytest.approx(10.0, rel=1e-6)
    apply_compensation(survey, model, smooth_hz=1.0)  # must not raise


def test_smooth_hz_without_a_usable_sample_rate_raises():
    theta_c, phi_c = _fibonacci_sphere(400, seed=15)
    model = build_calibration(_make_df(theta_c, phi_c), method="nearest")
    survey = _survey_with_compass_noise(n=500, seed=16).drop(columns=["Time"])

    with pytest.raises(ValueError, match="sample rate"):
        apply_compensation(survey, model, smooth_hz=1.0)


def test_smooth_hz_above_nyquist_raises():
    theta_c, phi_c = _fibonacci_sphere(400, seed=17)
    model = build_calibration(_make_df(theta_c, phi_c), method="nearest")
    survey = _survey_with_compass_noise(n=500, rate_hz=10.0, seed=18)

    with pytest.raises(ValueError, match="Nyquist"):
        apply_compensation(survey, model, smooth_hz=9.0)
