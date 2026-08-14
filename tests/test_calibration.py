import numpy as np
import pandas as pd
import pytest

from mag_heading.calibration import build_calibration, quality_report
from mag_heading.calibration import CalibrationModel, scan_calibration_candidates
from mag_heading.calibration import trim_segment_outliers, auto_build_calibration
from mag_heading.heading import unit_vector

BASELINE = 50000.0


def delta_b_true(theta, phi):
    return 2.0 * np.cos(theta) + 0.5 * np.sin(phi)


def _fibonacci_sphere(n, seed=0):
    """Roughly even coverage of the sphere, for deterministic interpolation tests."""
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


def test_build_calibration_baseline_and_delta_b():
    theta, phi = _fibonacci_sphere(100)
    df = _make_df(theta, phi)
    model = build_calibration(df)

    assert model.baseline == pytest.approx(np.median(df["Mag"]))
    np.testing.assert_allclose(
        model.delta_b_cal, df["Mag"].to_numpy() - model.baseline, atol=1e-8
    )


def test_predict_exact_at_training_points_both_methods():
    theta, phi = _fibonacci_sphere(150)
    df = _make_df(theta, phi)

    for method in ("nearest", "spline"):
        model = build_calibration(df, method=method)
        pred = model.predict(model.theta_cal, model.phi_cal)
        np.testing.assert_allclose(pred, model.delta_b_cal, atol=1e-6)


def test_spline_interpolates_smooth_function_better_than_nearest():
    theta_cal, phi_cal = _fibonacci_sphere(200)
    df_cal = _make_df(theta_cal, phi_cal)

    theta_q, phi_q = _fibonacci_sphere(200, seed=1)
    theta_q = theta_q + 0.01  # offset so query points differ from calibration points
    true_delta = delta_b_true(theta_q, phi_q)

    model_nearest = build_calibration(df_cal, method="nearest")
    model_spline = build_calibration(df_cal, method="spline")

    err_nearest = np.sqrt(np.mean((model_nearest.predict(theta_q, phi_q) - true_delta) ** 2))
    err_spline = np.sqrt(np.mean((model_spline.predict(theta_q, phi_q) - true_delta) ** 2))

    assert err_spline < err_nearest


def test_grid_binning_reveals_non_heading_noise():
    # Many repeated visits to a small cluster of headings, with injected noise that
    # has nothing to do with heading (e.g. field gradient/drone interference). Binning
    # must average these together so self-compensation residual reflects that noise,
    # rather than each raw point trivially recovering itself (distance-0 nearest match).
    rng = np.random.default_rng(42)
    n = 500
    theta = np.full(n, np.pi / 4) + rng.normal(scale=0.005, size=n)  # ~0.3 deg jitter
    phi = np.full(n, 0.5) + rng.normal(scale=0.005, size=n)
    df = _make_df(theta, phi, noise=0.5, seed=42)  # 0.5 nT non-heading noise

    model = build_calibration(df, method="nearest", grid_resolution_deg=5.0)

    assert model.bin_counts.max() > 1
    assert len(model.bin_delta_b) < n

    report = quality_report(model)
    assert report["self_compensation_std_nT"] > 0.1
    assert report["bins_with_single_sample_fraction"] < 1.0


def test_quality_report_self_compensation_near_zero():
    theta, phi = _fibonacci_sphere(120)
    df = _make_df(theta, phi)
    model = build_calibration(df, method="spline")

    report = quality_report(model, max_variation=1e-3)
    assert report["self_compensation_p2p_nT"] < 1e-3
    assert report["self_compensation_pass"] is True


def test_quality_report_heading_coverage_detects_gap():
    theta_cal, phi_cal = _fibonacci_sphere(200)
    upper_hemisphere = theta_cal < (np.pi / 2)
    df_cal = _make_df(theta_cal[upper_hemisphere], phi_cal[upper_hemisphere])
    model = build_calibration(df_cal, coverage_threshold_deg=5.0)

    theta_survey, phi_survey = _fibonacci_sphere(200, seed=2)
    df_survey = _make_df(theta_survey, phi_survey)

    report = quality_report(model, survey_df=df_survey)
    assert report["survey_uncovered_fraction"] > 0.2


def test_save_load_round_trip(tmp_path):
    theta, phi = _fibonacci_sphere(80)
    df = _make_df(theta, phi)
    model = build_calibration(
        df, method="spline", coverage_threshold_deg=7.5, metadata={"note": "unit test"}
    )

    path = tmp_path / "model.npz"
    model.save(path)
    loaded = CalibrationModel.load(path)

    assert loaded.method == "spline"
    assert loaded.coverage_threshold_deg == pytest.approx(7.5)
    assert loaded.metadata["note"] == "unit test"

    theta_q, phi_q = _fibonacci_sphere(30, seed=3)
    np.testing.assert_allclose(
        model.predict(theta_q, phi_q), loaded.predict(theta_q, phi_q), atol=1e-8
    )


def test_invalid_method_raises():
    theta, phi = _fibonacci_sphere(10)
    df = _make_df(theta, phi)
    with pytest.raises(ValueError):
        build_calibration(df, method="bogus")


def _random_heading_segment(n, seed, theta_max=np.pi / 2):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.05, theta_max, size=n)
    phi = rng.uniform(-np.pi, np.pi, size=n)
    return theta, phi


def test_scan_ranks_clean_segment_above_noisy_segment():
    # A clean, low-noise calibration window followed by a noisy one (e.g. flown
    # over a field gradient or near drone interference) -- scan should surface
    # windows inside the clean region as the best candidates.
    theta_clean, phi_clean = _random_heading_segment(300, seed=10)
    df_clean = _make_df(theta_clean, phi_clean, noise=0.05, seed=10)

    theta_noisy, phi_noisy = _random_heading_segment(300, seed=11)
    df_noisy = _make_df(theta_noisy, phi_noisy, noise=3.0, seed=11)
    # keep the same background level so windows spanning the boundary aren't
    # trivially flagged by a baseline jump rather than by noise.
    df_noisy["Mag"] += BASELINE - df_noisy["Mag"].median() + df_clean["Mag"].median() - BASELINE

    df = pd.concat([df_clean, df_noisy], ignore_index=True)

    candidates = scan_calibration_candidates(df, window_size=100, step=50, top_n=20)
    assert len(candidates) > 0

    best = candidates[0]
    assert best["end"] < 300  # entirely within the clean region

    clean_p2p = [c["self_compensation_p2p_nT"] for c in candidates if c["end"] < 300]
    noisy_p2p = [c["self_compensation_p2p_nT"] for c in candidates if c["start"] >= 300]
    assert clean_p2p and noisy_p2p
    assert max(clean_p2p) < min(noisy_p2p)


def test_scan_rejects_window_larger_than_data():
    theta, phi = _fibonacci_sphere(50)
    df = _make_df(theta, phi)
    with pytest.raises(ValueError):
        scan_calibration_candidates(df, window_size=len(df) + 1)


def test_trim_outliers_keeps_lone_important_heading():
    # A dense, repeated heading (many observations, one of which is a spike) plus a
    # single lone sample at a completely different heading that also looks extreme.
    # The spike in the dense cluster is redundant noise and should go; the lone
    # point is the only evidence for its heading and must survive even though it
    # looks just as extreme, or compensation would have a coverage gap right there.
    rng = np.random.default_rng(1)
    n_cluster = 10
    # jitter small enough that all cluster points land in the same 5deg grid bin
    theta_cluster = np.full(n_cluster, 0.8) + rng.normal(scale=0.001, size=n_cluster)
    phi_cluster = np.full(n_cluster, 0.8) + rng.normal(scale=0.001, size=n_cluster)
    mag_cluster = np.full(n_cluster, BASELINE) + rng.normal(scale=0.05, size=n_cluster)
    mag_cluster[3] += 20.0

    theta_lone = np.array([2.0])
    phi_lone = np.array([-2.0])
    mag_lone = np.array([BASELINE + 20.0])

    theta = np.concatenate([theta_cluster, theta_lone])
    phi = np.concatenate([phi_cluster, phi_lone])
    mag = np.concatenate([mag_cluster, mag_lone])
    vec = unit_vector(theta, phi)
    df = pd.DataFrame(
        {
            "id": np.arange(len(theta)),
            "Mag": mag,
            "CompassX": vec[:, 0],
            "CompassY": vec[:, 1],
            "CompassZ": vec[:, 2],
        }
    )

    trimmed, stats = trim_segment_outliers(df, grid_resolution_deg=5.0, mad_threshold=3.0)

    assert stats["n_removed"] >= 1
    surviving_ids = set(trimmed["id"])
    assert 3 not in surviving_ids  # the spike, redundant within its well-populated bin
    assert n_cluster in surviving_ids  # the lone point, sole evidence for its heading


def test_trim_never_empties_a_bin():
    rng = np.random.default_rng(3)
    n = 4
    # jitter small enough that all 4 points land in the same 5deg grid bin
    theta = np.full(n, 1.0) + rng.normal(scale=0.001, size=n)
    phi = np.full(n, 1.0) + rng.normal(scale=0.001, size=n)
    mag = BASELINE + np.array([-50.0, -10.0, 10.0, 50.0])  # every point looks extreme
    vec = unit_vector(theta, phi)
    df = pd.DataFrame(
        {"Mag": mag, "CompassX": vec[:, 0], "CompassY": vec[:, 1], "CompassZ": vec[:, 2]}
    )

    trimmed, stats = trim_segment_outliers(
        df, grid_resolution_deg=5.0, mad_threshold=0.1, max_trim_fraction_per_bin=0.5
    )
    assert stats["n_after"] >= 2
    assert len(trimmed) == stats["n_after"]


def test_build_calibration_trim_outliers_improves_p2p():
    rng = np.random.default_rng(2)
    n = 200
    theta = np.full(n, 0.9) + rng.normal(scale=0.02, size=n)
    phi = np.full(n, -0.4) + rng.normal(scale=0.02, size=n)
    mag = np.full(n, BASELINE) + rng.normal(scale=0.1, size=n)
    spike_idx = rng.choice(n, size=10, replace=False)
    mag[spike_idx] += rng.choice([-1.0, 1.0], size=10) * 15.0

    vec = unit_vector(theta, phi)
    df = pd.DataFrame(
        {"Mag": mag, "CompassX": vec[:, 0], "CompassY": vec[:, 1], "CompassZ": vec[:, 2]}
    )

    model_raw = build_calibration(df, method="nearest", trim_outliers=False)
    model_trimmed = build_calibration(df, method="nearest", trim_outliers=True)

    report_raw = quality_report(model_raw)
    report_trimmed = quality_report(model_trimmed)

    assert report_trimmed["self_compensation_p2p_nT"] < report_raw["self_compensation_p2p_nT"]
    assert model_trimmed.metadata["outlier_trim_stats"]["n_removed"] > 0


def test_quality_report_dominant_heading_diagnostic():
    rng = np.random.default_rng(4)
    n_dom = 300
    theta_dom = np.full(n_dom, 0.6) + rng.normal(scale=0.03, size=n_dom)
    phi_dom = np.full(n_dom, 1.2) + rng.normal(scale=0.03, size=n_dom)

    n_minor = 50
    theta_minor = np.full(n_minor, 1.3) + rng.normal(scale=0.05, size=n_minor)
    phi_minor = np.full(n_minor, -2.5) + rng.normal(scale=0.05, size=n_minor)

    theta_survey = np.concatenate([theta_dom, theta_minor])
    phi_survey = np.concatenate([phi_dom, phi_minor])
    df_survey = _make_df(theta_survey, phi_survey, noise=0.05, seed=4)

    df_cal_covering = _make_df(theta_dom[:100], phi_dom[:100], noise=0.05, seed=5)
    report_covering = quality_report(build_calibration(df_cal_covering), survey_df=df_survey)
    assert report_covering["survey_dominant_heading_covered"] is True

    df_cal_missing = _make_df(theta_minor, phi_minor, noise=0.05, seed=6)
    report_missing = quality_report(build_calibration(df_cal_missing), survey_df=df_survey)
    assert report_missing["survey_dominant_heading_covered"] is False


def test_scan_prioritizes_dominant_heading_coverage():
    # A window with only mediocre self-compensation but that covers the survey's
    # dominant heading should still outrank a window with excellent self-compensation
    # that covers only a rare, unrepresentative heading -- missing the dominant
    # (primary flight-line) heading matters more than raw flatness.
    rng = np.random.default_rng(7)
    n_dom = 400
    theta_dom = np.full(n_dom, 0.7) + rng.normal(scale=0.05, size=n_dom)
    phi_dom = np.full(n_dom, 0.3) + rng.normal(scale=0.05, size=n_dom)
    df_dom = _make_df(theta_dom, phi_dom, noise=0.3, seed=7)

    n_clean = 100
    theta_clean = np.full(n_clean, 1.4) + rng.normal(scale=0.02, size=n_clean)
    phi_clean = np.full(n_clean, -2.8) + rng.normal(scale=0.02, size=n_clean)
    df_clean = _make_df(theta_clean, phi_clean, noise=0.01, seed=8)

    df = pd.concat([df_dom, df_clean], ignore_index=True)

    candidates = scan_calibration_candidates(df, window_size=100, step=100, top_n=20)

    clean_candidates = [c for c in candidates if c["start"] >= n_dom]
    dom_candidates = [c for c in candidates if c["end"] < n_dom]
    assert clean_candidates and dom_candidates
    assert min(c["self_compensation_p2p_nT"] for c in clean_candidates) < min(
        c["self_compensation_p2p_nT"] for c in dom_candidates
    )

    assert candidates[0]["dominant_heading_covered"] is True


def test_auto_build_calibration_stops_after_first_passing_candidate():
    # The whole file already has uniform angular diversity everywhere (Fibonacci
    # sphere), so with a lenient coverage target the very first ranked window
    # should already pass on its own -- auto_build must not keep bolting on
    # unnecessary extra segments once the stopping criteria are met.
    theta, phi = _fibonacci_sphere(500, seed=40)
    df = _make_df(theta, phi, noise=0.02, seed=40)

    result = auto_build_calibration(
        df, window_size=200, step=100, method="nearest",
        max_variation=1.0, coverage_target=0.5, max_candidates=20,
    )

    assert result["passed"] is True
    assert result["model"] is not None
    assert len(result["segments"]) == 1


def test_auto_build_calibration_combines_complementary_segments_to_reach_target():
    # Four survey "legs" at well-separated headings (like four boustrophedon
    # lines running in different compass directions), each internally flat/clean.
    # window_size is smaller than a leg, so no single window can ever see more
    # than two adjacent legs -- reaching a strict coverage target requires
    # auto_build to accept segments from more than one leg boundary.
    rng = np.random.default_rng(20)
    leg_len = 1000
    centers = [(0.7, 0.3), (0.7, 2.0), (0.7, -2.0), (0.7, -1.0)]
    thetas, phis = [], []
    for theta_c, phi_c in centers:
        thetas.append(np.full(leg_len, theta_c) + rng.normal(scale=0.03, size=leg_len))
        phis.append(np.full(leg_len, phi_c) + rng.normal(scale=0.03, size=leg_len))
    theta = np.concatenate(thetas)
    phi = np.concatenate(phis)
    df = _make_df(theta, phi, noise=0.05, seed=20)

    result = auto_build_calibration(
        df, window_size=400, step=200, method="nearest",
        max_variation=1.0, coverage_target=0.05, max_candidates=30,
    )

    assert result["passed"] is True
    assert result["model"] is not None
    assert len(result["segments"]) >= 2

    leg_ranges = [(i * leg_len, i * leg_len + leg_len - 1) for i in range(4)]

    def touches(seg, leg):
        return seg[0] <= leg[1] and seg[1] >= leg[0]

    for leg in leg_ranges:
        assert any(touches(seg, leg) for seg in result["segments"]), \
            f"no accepted segment touches leg {leg}"


def test_build_calibration_drops_nonfinite_samples():
    # A single NaN makes np.median return NaN, which turns baseline, every
    # delta_b and the resulting p2p into NaN -- and a NaN p2p then passes every
    # `p2p > threshold` gate, since comparisons against NaN are always False.
    # Filtering at the source is what keeps those gates meaningful.
    theta, phi = _fibonacci_sphere(200)
    df = _make_df(theta, phi, noise=0.02, seed=70)
    df.loc[5, "Mag"] = np.nan
    df.loc[9, "CompassY"] = np.nan

    model = build_calibration(df, method="nearest")

    assert model.metadata["n_nonfinite_dropped"] == 2
    assert len(model.mag_cal) == len(df) - 2
    assert np.isfinite(model.baseline)
    assert np.isfinite(quality_report(model)["self_compensation_p2p_nT"])


def test_auto_build_does_not_accept_a_nan_segment():
    # Every candidate window contains a NaN, so before the fix auto-build
    # accepted an all-NaN model with zero p2p rejections.
    theta, phi = _fibonacci_sphere(900, seed=71)
    df = _make_df(theta, phi, noise=0.02, seed=71)
    df.loc[::120, "Mag"] = np.nan

    result = auto_build_calibration(
        df, window_size=(300,), method="nearest", max_variation=0.5, coverage_target=0.05
    )

    assert result["model"] is not None
    assert np.isfinite(result["model"].baseline)
    assert np.isfinite(result["report"]["self_compensation_p2p_nT"])
    assert result["model"].metadata["n_nonfinite_dropped"] > 0


def test_calibration_model_rejects_nonfinite_input_directly():
    theta, phi = _fibonacci_sphere(50)
    df = _make_df(theta, phi)
    mag = df["Mag"].to_numpy().copy()
    mag[3] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        CalibrationModel(
            mag_cal=mag,
            compass_x_cal=df["CompassX"].to_numpy(),
            compass_y_cal=df["CompassY"].to_numpy(),
            compass_z_cal=df["CompassZ"].to_numpy(),
        )


def test_spline_reports_actionable_error_when_too_few_grid_bins():
    # A near-constant heading collapses into 1-2 angular bins; the thin-plate
    # spline needs 4 in 3D. Without the explicit check this surfaced as a raw
    # scipy "At least 4 data points are required" traceback out of the CLI.
    rng = np.random.default_rng(72)
    n = 200
    theta = np.full(n, 0.8) + rng.normal(scale=0.001, size=n)
    phi = np.full(n, 0.5) + rng.normal(scale=0.001, size=n)
    df = _make_df(theta, phi, noise=0.05, seed=72)

    assert build_calibration(df, method="nearest") is not None

    with pytest.raises(ValueError, match="at least 4 angular grid bins"):
        build_calibration(df, method="spline")


def test_predict_and_coverage_tolerate_nonfinite_survey_headings():
    # cKDTree.query raises on non-finite input, so one dropped compass reading
    # used to abort compensation of the whole survey file.
    theta, phi = _fibonacci_sphere(200)
    model = build_calibration(_make_df(theta, phi, noise=0.02, seed=73), method="spline")

    q_theta = np.array([0.8, np.nan, 1.2])
    q_phi = np.array([0.3, 0.4, np.nan])

    predicted = model.predict(q_theta, q_phi)
    assert np.isfinite(predicted[0])
    assert np.isnan(predicted[1]) and np.isnan(predicted[2])

    # a heading-less row is certainly not covered, so it must not read as 0 distance
    chord = model.coverage_distance(q_theta, q_phi)
    assert np.isfinite(chord[0])
    assert np.isinf(chord[1]) and np.isinf(chord[2])


def test_auto_build_accepts_numpy_integer_window_size():
    # numpy scalars are not subclasses of int, so the sequence branch used to
    # run and raise "'numpy.int64' object is not iterable".
    theta, phi = _fibonacci_sphere(400, seed=74)
    df = _make_df(theta, phi, noise=0.02, seed=74)

    result = auto_build_calibration(
        df, window_size=np.int64(200), method="nearest", max_variation=1.0, coverage_target=0.5
    )
    assert result["window_size"] == 200


def test_coverage_subsampling_does_not_change_the_reported_result():
    # The search may estimate coverage from a subsample, but what gets reported
    # (and gates saving) must be the exact figure over the whole survey.
    theta, phi = _fibonacci_sphere(1500, seed=75)
    df = _make_df(theta, phi, noise=0.02, seed=75)

    exact = auto_build_calibration(
        df, window_size=(300,), method="nearest", max_variation=1.0,
        coverage_target=0.3, coverage_sample_limit=0,
    )
    sampled = auto_build_calibration(
        df, window_size=(300,), method="nearest", max_variation=1.0,
        coverage_target=0.3, coverage_sample_limit=200,
    )

    assert sampled["segments"] == exact["segments"]
    assert sampled["passed"] == exact["passed"]
    assert sampled["report"]["survey_uncovered_fraction"] == pytest.approx(
        exact["report"]["survey_uncovered_fraction"]
    )
    assert sampled["report"]["self_compensation_p2p_nT"] == pytest.approx(
        exact["report"]["self_compensation_p2p_nT"]
    )


def test_scan_coverage_subsampling_keeps_ranking():
    # Subsampling only affects survey_uncovered_fraction, which is the *third*
    # sort key. The primary ranking (dominant-heading coverage, then self-comp
    # p2p) is computed exactly and must be untouched; only windows whose p2p ties
    # may swap on the noisier coverage estimate, which is why the ordering is
    # asserted on p2p rather than on start positions.
    theta, phi = _fibonacci_sphere(1200, seed=76)
    df = _make_df(theta, phi, noise=0.05, seed=76)

    exact = scan_calibration_candidates(df, window_size=200, step=100, coverage_sample_limit=0)
    sampled = scan_calibration_candidates(df, window_size=200, step=100, coverage_sample_limit=300)

    assert {c["start"] for c in sampled} == {c["start"] for c in exact}
    assert sampled[0]["start"] == exact[0]["start"]
    assert [c["self_compensation_p2p_nT"] for c in sampled] == [
        c["self_compensation_p2p_nT"] for c in exact
    ]

    exact_by_start = {c["start"]: c["survey_uncovered_fraction"] for c in exact}
    for c in sampled:
        assert c["survey_uncovered_fraction"] == pytest.approx(
            exact_by_start[c["start"]], abs=0.05
        )


def test_auto_build_calibration_refuses_when_no_candidate_qualifies():
    # Non-heading noise this large can't be explained away by per-bin outlier
    # trimming (it's not a handful of spikes, it's the whole distribution), so
    # no window/combination should ever satisfy a tight max_variation -- this
    # must come back as passed=False rather than silently handing back a bad
    # model for the caller to save.
    theta, phi = _fibonacci_sphere(2000, seed=30)
    df = _make_df(theta, phi, noise=5.0, seed=30)

    result = auto_build_calibration(
        df, window_size=300, step=150, method="nearest",
        max_variation=0.3, coverage_target=0.1, max_candidates=10,
    )

    assert result["passed"] is False
