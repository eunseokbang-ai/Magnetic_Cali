import numpy as np
import pandas as pd
import pytest

from mag_heading.lines import detect_lines, direction_groups, estimate_line_spacing_m
from mag_heading.line_leveling import compute_line_leveling, apply_line_leveling

EARTH_RADIUS_M = 6371000.0
LAT0, LON0 = 34.5, 126.5
SPACING = 50.0
LINE_LENGTH = 600.0


def _xy_to_latlon(x, y):
    lat = LAT0 + np.degrees(y / EARTH_RADIUS_M)
    lon = LON0 + np.degrees(x / (EARTH_RADIUS_M * np.cos(np.radians(LAT0))))
    return lat, lon


def _geology(x, y):
    """Smooth structure well above line-spacing scale -- what the
    correction must leave alone."""
    return 20.0 * np.sin(2 * np.pi * x / 900.0) + 15.0 * np.cos(2 * np.pi * y / 700.0)


def _turn_points(x0, y0, x1, y1, n=25, loop_out=30.0):
    """See tests/test_lines.py's copy for why this matters: detect_lines
    needs a genuine off-azimuth stretch between lines, not a teleport."""
    t = np.linspace(0.0, 1.0, n)
    lx = x0 + (x1 - x0) * t
    ly = y0 + (y1 - y0) * t
    dx, dy = x1 - x0, y1 - y0
    length = max(float(np.hypot(dx, dy)), 1e-6)
    perp = np.array([-dy, dx]) / length
    bulge = loop_out * np.sin(np.pi * t)
    return lx + perp[0] * bulge, ly + perp[1] * bulge


def _survey(n_lines, offset_nt, n_per_line=150, seed=0):
    """Boustrophedon lines with real geology plus a constant forward/reverse
    step (the thing this module has to find and remove), joined by curved
    turns so detect_lines can find the line breaks."""
    rng = np.random.default_rng(seed)
    xs, ys, values = [], [], []
    prev_end = None
    for i in range(n_lines):
        x = np.full(n_per_line, float(i * SPACING))
        y = np.linspace(0.0, LINE_LENGTH, n_per_line)
        reverse = i % 2 == 1
        if reverse:
            y = y[::-1]
        if prev_end is not None:
            tx, ty = _turn_points(prev_end[0], prev_end[1], x[0], y[0])
            xs.append(tx[1:-1])
            ys.append(ty[1:-1])
            values.append(np.full(len(tx) - 2, np.nan))
        value = _geology(x, y) + rng.normal(scale=0.05, size=n_per_line)
        value = value + (offset_nt / 2.0 if reverse else -offset_nt / 2.0)
        xs.append(x)
        ys.append(y)
        values.append(value)
        prev_end = (x[-1], y[-1])
    x0 = np.concatenate(xs)
    y0 = np.concatenate(ys)
    value = np.concatenate(values)
    lat, lon = _xy_to_latlon(x0, y0)
    return pd.DataFrame({"Latitude": lat, "Longitude": lon, "value": value})


def _leveled(n_lines=10, offset_nt=6.0, **kw):
    df = _survey(n_lines, offset_nt, **kw)
    lined = detect_lines(df)
    az = lined.attrs["dominant_azimuth_deg"]
    spacing = estimate_line_spacing_m(lined, az)
    result = compute_line_leveling(lined, "value", az, spacing)
    return lined, result, az


def test_recovers_the_injected_forward_reverse_offset():
    lined, result, _az = _leveled(n_lines=10, offset_nt=6.0)
    assert result.applied, result.reason
    # sign convention: offset_nt is (group A - group B); the survey was
    # built with B (reverse) high by offset/2 and A low by offset/2, i.e.
    # A - B = -offset_nt as constructed.
    assert result.offset_nt == pytest.approx(-6.0, abs=0.6)
    assert result.offset_spread_nt < abs(result.offset_nt)


def test_applying_the_correction_removes_the_offset():
    lined, result, az = _leveled(n_lines=10, offset_nt=6.0)
    corrected = apply_line_leveling(lined, "value", result)
    lined["corrected"] = corrected

    groups = direction_groups(lined, az)
    a_lines = [lid for lid, g in groups.items() if g == "A"]
    b_lines = [lid for lid, g in groups.items() if g == "B"]

    def _mean_for(col, ids):
        return lined[lined.line_id.isin(ids)][col].mean()

    before_gap = abs(_mean_for("value", a_lines) - _mean_for("value", b_lines))
    after_gap = abs(_mean_for("corrected", a_lines) - _mean_for("corrected", b_lines))
    assert after_gap < before_gap * 0.3


def test_declines_with_too_few_lines():
    lined, result, _az = _leveled(n_lines=2, offset_nt=6.0)
    assert not result.applied
    assert "측선" in result.reason


def test_declines_when_all_lines_share_one_direction():
    # Every line flown the same way (a "ferry" survey), joined by realistic
    # curved turns so detect_lines still separates them into distinct lines.
    rng = np.random.default_rng(1)
    xs, ys, values = [], [], []
    prev_end = None
    for i in range(4):
        x = np.full(150, float(i * SPACING))
        y = np.linspace(0.0, LINE_LENGTH, 150)
        if prev_end is not None:
            tx, ty = _turn_points(prev_end[0], prev_end[1], x[0], y[0])
            xs.append(tx[1:-1])
            ys.append(ty[1:-1])
            values.append(np.full(len(tx) - 2, np.nan))
        value = _geology(x, y) + rng.normal(scale=0.05, size=150)
        xs.append(x)
        ys.append(y)
        values.append(value)
        prev_end = (x[-1], y[-1])
    x0, y0, value = np.concatenate(xs), np.concatenate(ys), np.concatenate(values)
    lat, lon = _xy_to_latlon(x0, y0)
    one_dir = pd.DataFrame({"Latitude": lat, "Longitude": lon, "value": value})

    lined2 = detect_lines(one_dir)
    line_ids = sorted(int(v) for v in lined2["line_id"].unique() if v >= 0)
    # Exact count isn't the point here (see test_detect_lines_separates_
    # same_direction_passes for that) -- just enough distinct lines that
    # "all one direction" is a meaningful precondition to decline on.
    assert len(line_ids) >= 3, "test setup: same-direction lines must be detected first"

    az = lined2.attrs["dominant_azimuth_deg"]
    result = compute_line_leveling(lined2, "value", az, 50.0)
    assert not result.applied
    assert "반대 방향" in result.reason


def test_a_clean_survey_with_no_offset_yields_a_near_zero_correction():
    lined, result, az = _leveled(n_lines=10, offset_nt=0.0, seed=3)
    assert result.applied
    assert abs(result.offset_nt) < 1.0


def test_apply_is_a_no_op_when_not_applied():
    lined, _r, _az = _leveled(n_lines=2, offset_nt=6.0)
    declined = compute_line_leveling(lined, "value", 90.0, 50.0)
    out = apply_line_leveling(lined, "value", declined)
    np.testing.assert_array_equal(out, lined["value"].to_numpy())
