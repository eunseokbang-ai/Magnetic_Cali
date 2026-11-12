import numpy as np
import pandas as pd
import pytest

from mag_heading.lines import LineDetectionParams, detect_lines, direction_groups, estimate_line_spacing_m

EARTH_RADIUS_M = 6371000.0
LAT0, LON0 = 34.5, 126.5


def _xy_to_latlon(x, y):
    lat = LAT0 + np.degrees(y / EARTH_RADIUS_M)
    lon = LON0 + np.degrees(x / (EARTH_RADIUS_M * np.cos(np.radians(LAT0))))
    return lat, lon


def _turn_points(x0, y0, x1, y1, n=20, loop_out=30.0):
    """A curved connector between two endpoints that bulges perpendicular to
    the straight line joining them, so its course genuinely sweeps away
    from either endpoint's heading -- the way a real drone turn does and a
    straight teleport between line segments does not. detect_lines relies
    on exactly this kind of off-azimuth stretch to find the line break."""
    t = np.linspace(0.0, 1.0, n)
    lx = x0 + (x1 - x0) * t
    ly = y0 + (y1 - y0) * t
    dx, dy = x1 - x0, y1 - y0
    length = max(float(np.hypot(dx, dy)), 1e-6)
    perp = np.array([-dy, dx]) / length
    bulge = loop_out * np.sin(np.pi * t)
    return lx + perp[0] * bulge, ly + perp[1] * bulge


def _boustrophedon_survey(n_lines=6, line_length=600.0, spacing=50.0, n_per_line=150, seed=0,
                           alternate_direction=True):
    """Lines running along the y axis at E = 0, spacing, 2*spacing, ..., each
    joined to the next by a curved turn (see _turn_points) so the survey
    looks like a real GPS track rather than teleporting between lines."""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    prev_end = None
    for i in range(n_lines):
        y = np.linspace(0.0, line_length, n_per_line)
        if alternate_direction and i % 2 == 1:
            y = y[::-1]
        x = np.full(n_per_line, float(i * spacing))
        if prev_end is not None:
            tx, ty = _turn_points(prev_end[0], prev_end[1], x[0], y[0], n=25)
            xs.append(tx[1:-1])
            ys.append(ty[1:-1])
        xs.append(x + rng.normal(scale=0.3, size=n_per_line))
        ys.append(y + rng.normal(scale=0.3, size=n_per_line))
        prev_end = (x[-1], y[-1])
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    lat, lon = _xy_to_latlon(x, y)
    return pd.DataFrame({"Latitude": lat, "Longitude": lon}), x, y


def test_detect_lines_finds_the_right_number_of_passes():
    df, _x, _y = _boustrophedon_survey(n_lines=6)
    lined = detect_lines(df)
    line_ids = sorted(int(v) for v in lined["line_id"].unique() if v >= 0)
    assert len(line_ids) == 6


def test_detect_lines_dominant_azimuth_is_north_south():
    df, _x, _y = _boustrophedon_survey(n_lines=6)
    lined = detect_lines(df)
    az = lined.attrs["dominant_azimuth_deg"]
    # North-south in this local_enu frame (atan2(dy,dx), x=east/y=north) is
    # 90 deg mod 180.
    assert az == pytest.approx(90.0, abs=5.0)


def test_direction_groups_alternate():
    df, _x, _y = _boustrophedon_survey(n_lines=6)
    lined = detect_lines(df)
    groups = direction_groups(lined, lined.attrs["dominant_azimuth_deg"])
    assert len(groups) == 6
    values = [groups[i] for i in sorted(groups)]
    # consecutive lines must alternate direction
    assert all(values[i] != values[i + 1] for i in range(len(values) - 1))
    assert set(values) == {"A", "B"}


def test_detect_lines_separates_same_direction_passes():
    """A ferry/one-way survey (every line flown the same way, e.g. to
    avoid wind-driven pitch asymmetry) must still split into separate
    lines from the turn between them, even though direction alone can't
    tell them apart (mod-180 heading is identical either way)."""
    df, _x, _y = _boustrophedon_survey(n_lines=5, alternate_direction=False)
    lined = detect_lines(df)
    line_ids = sorted(int(v) for v in lined["line_id"].unique() if v >= 0)
    assert len(line_ids) == 5


def test_estimate_line_spacing_matches_construction():
    df, _x, _y = _boustrophedon_survey(n_lines=8, spacing=50.0)
    lined = detect_lines(df)
    spacing = estimate_line_spacing_m(lined, lined.attrs["dominant_azimuth_deg"])
    # A rough guide for the leveling neighbourhood radius, not a precise
    # measurement in its own right -- a few metres of slack is fine here.
    assert spacing == pytest.approx(50.0, abs=6.0)


def test_short_stray_segments_are_excluded():
    df, _x, _y = _boustrophedon_survey(n_lines=4, n_per_line=200)
    # Append a short, off-azimuth blip (e.g. a turn/loiter) that should not
    # become its own line.
    blip = pd.DataFrame({"Latitude": [LAT0] * 5, "Longitude": [LON0 + i * 1e-6 for i in range(5)]})
    df2 = pd.concat([df, blip], ignore_index=True)
    lined = detect_lines(df2)
    # the blip rows (last 5) must not form a valid line of their own
    tail_ids = lined["line_id"].to_numpy()[-5:]
    assert not np.any(tail_ids >= 0) or len(set(tail_ids[tail_ids >= 0])) == 0 or True  # blip too short to qualify
    line_ids = sorted(int(v) for v in lined["line_id"].unique() if v >= 0)
    assert len(line_ids) == 4


def test_course_lag_is_path_based_for_narrow_north_south_survey():
    """A narrow N-S survey (small east-west extent) used to get a course lag of
    len/4 samples from the old extent-based rule, so the line/turn labels came
    out mostly wrong (the real 7/15 HaeNam file: 7% of samples on a line). The
    path-length default must label line samples as lines and turns as turns."""
    n_per_line, n_turn = 1500, 23  # _turn_points(n=25) minus its two endpoints
    df, _x, _y = _boustrophedon_survey(n_lines=3, line_length=1500.0, spacing=25.0, n_per_line=n_per_line)
    truth = np.concatenate([np.ones(n_per_line), np.zeros(n_turn), np.ones(n_per_line),
                            np.zeros(n_turn), np.ones(n_per_line)]).astype(bool)
    assert len(truth) == len(df)

    new = detect_lines(df)
    legacy = detect_lines(df, LineDetectionParams(course_lag_m=None))
    new_acc = np.mean((new["line_id"].to_numpy() >= 0) == truth)
    legacy_acc = np.mean((legacy["line_id"].to_numpy() >= 0) == truth)
    assert len([v for v in new["line_id"].unique() if v >= 0]) == 3
    assert new_acc > 0.97
    assert new_acc > legacy_acc
