import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from mag_heading.calibration import build_calibration
from mag_heading.heading import unit_vector
from mag_heading.plot import (
    grid_interpolate,
    local_enu,
    plot_2d_comparison,
    plot_calibration_quality,
    plot_compensation_timeseries,
    plot_heading_timeseries,
)


def test_local_enu_centers_on_mean():
    lat = np.array([34.560, 34.561, 34.562])
    lon = np.array([126.350, 126.351, 126.352])
    x, y = local_enu(lat, lon)
    assert x.mean() == pytest.approx(0.0, abs=1e-6)
    assert y.mean() == pytest.approx(0.0, abs=1e-6)


def test_local_enu_known_distance():
    # 1 degree of latitude on a sphere of Earth's mean radius (6371 km) is ~111.19 km;
    # this is a spherical approximation, not the WGS84 ellipsoid value (~110.6-111.7 km).
    lat = np.array([0.0, 1.0])
    lon = np.array([0.0, 0.0])
    x, y = local_enu(lat, lon, lat0=0.0, lon0=0.0)
    expected_m_per_deg = np.radians(1.0) * 6371000.0
    assert y[1] - y[0] == pytest.approx(expected_m_per_deg, rel=1e-6)
    assert x[0] == pytest.approx(0.0, abs=1e-6)


def test_grid_interpolate_recovers_linear_function():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 100, size=200)
    y = rng.uniform(0, 100, size=200)
    values = 2.0 * x + 3.0 * y  # smooth, easy for cubic/linear to recover

    gx, gy, gz = grid_interpolate(x, y, values, resolution_m=5.0, method="linear")
    assert gx.shape == gy.shape == gz.shape

    # spot-check the interior (away from the convex-hull edge) against the true function
    mid = gz.shape[0] // 2, gz.shape[1] // 2
    expected = 2.0 * gx[mid] + 3.0 * gy[mid]
    assert gz[mid] == pytest.approx(expected, rel=0.05)
    assert not np.any(np.isnan(gz))


def test_grid_interpolate_requires_min_points():
    with pytest.raises(ValueError):
        grid_interpolate([0, 1], [0, 1], [0, 1])


def _make_survey_df(n=50, seed=0):
    rng = np.random.default_rng(seed)
    lat = 34.56 + np.linspace(0, 0.001, n)
    lon = 126.35 + np.linspace(0, 0.001, n)
    theta = rng.uniform(0.1, 1.4, size=n)
    phi = rng.uniform(-np.pi, np.pi, size=n)
    vec = unit_vector(theta, phi)
    mag = 50000.0 + rng.normal(scale=1.0, size=n)
    return pd.DataFrame(
        {
            "Latitude": lat,
            "Longitude": lon,
            "Mag": mag,
            "MagComp": mag - rng.normal(scale=0.2, size=n),
            "CompassX": vec[:, 0],
            "CompassY": vec[:, 1],
            "CompassZ": vec[:, 2],
        }
    )


def test_plot_2d_comparison_two_panels_with_comp_col():
    df = _make_survey_df()
    fig = plot_2d_comparison(df, raw_col="Mag", comp_col="MagComp", resolution_m=10.0)
    assert len(fig.axes) >= 2


def test_plot_2d_comparison_single_panel_without_comp_col():
    df = _make_survey_df()
    fig = plot_2d_comparison(df, raw_col="Mag", comp_col=None, resolution_m=10.0)
    assert len(fig.axes) == 2  # 1 map axis + 1 colorbar axis


def test_plot_heading_timeseries_returns_figure_with_two_lines():
    df = _make_survey_df()
    fig = plot_heading_timeseries(df)
    ax = fig.axes[0]
    assert len(ax.lines) == 2


def test_plot_calibration_quality_returns_figure():
    df = _make_survey_df(n=80)
    model = build_calibration(df, method="nearest")
    fig = plot_calibration_quality(model)
    assert len(fig.axes) >= 2  # primary + twinx


def test_plot_compensation_timeseries_uses_sample_index_without_time_column():
    df = _make_survey_df()
    fig = plot_compensation_timeseries(df)
    ax = fig.axes[0]
    assert ax.get_xlabel() == "sample index"
    assert len(ax.lines) == 2  # raw + compensated


def test_plot_compensation_timeseries_uses_time_column_when_present():
    df = _make_survey_df(n=20)
    df["Time"] = [f"00:00:{i:02d}.000" for i in range(20)]
    fig = plot_compensation_timeseries(df)
    ax = fig.axes[0]
    assert ax.get_xlabel() == "time (s)"
    x_data = ax.lines[0].get_xdata()
    np.testing.assert_allclose(x_data, np.arange(20), atol=1e-6)


def test_plot_compensation_timeseries_single_line_without_comp_col():
    df = _make_survey_df()
    fig = plot_compensation_timeseries(df, comp_col=None)
    ax = fig.axes[0]
    assert len(ax.lines) == 1


def test_plot_compensation_timeseries_single_line_when_comp_col_missing():
    df = _make_survey_df().drop(columns=["MagComp"])
    fig = plot_compensation_timeseries(df)
    ax = fig.axes[0]
    assert len(ax.lines) == 1


def test_plot_heading_coverage_marks_uncovered_survey_headings():
    from mag_heading.calibration import build_calibration
    from mag_heading.plot import plot_heading_coverage

    df = _make_survey_df(n=200)
    model = build_calibration(df.iloc[:60].reset_index(drop=True))
    fig = plot_heading_coverage(model, survey_df=df)
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert any("NOT covered" in s for s in labels)
    assert any("calibration" in s for s in labels)
    fig_no_survey = plot_heading_coverage(model)
    assert len(fig_no_survey.axes[0].collections) == 1
