import numpy as np
import pandas as pd
import pytest

from mag_heading import cli
from mag_heading.io import read_pre_csv, write_pre_csv

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
    correction must leave alone. Same shape as test_line_leveling.py's."""
    return 20.0 * np.sin(2 * np.pi * x / 900.0) + 15.0 * np.cos(2 * np.pi * y / 700.0)


def _turn_points(x0, y0, x1, y1, n=25, loop_out=30.0):
    """See tests/test_lines.py for why a curved connector (not a straight
    teleport) is required for detect_lines to find real line breaks."""
    t = np.linspace(0.0, 1.0, n)
    lx = x0 + (x1 - x0) * t
    ly = y0 + (y1 - y0) * t
    dx, dy = x1 - x0, y1 - y0
    length = max(float(np.hypot(dx, dy)), 1e-6)
    perp = np.array([-dy, dx]) / length
    bulge = loop_out * np.sin(np.pi * t)
    return lx + perp[0] * bulge, ly + perp[1] * bulge


def _survey_df(n_lines=8, offset_nt=6.0, n_per_line=150, seed=0):
    """A boustrophedon survey with real geology plus a constant
    forward/reverse offset, as a full -pre.csv-shaped DataFrame (one
    combined track -- callers split it into files themselves so that the
    turns between files stay realistic, matching how a real multi-file
    survey is flown)."""
    rng = np.random.default_rng(seed)
    xs, ys, comp = [], [], []
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
            comp.append(np.full(len(tx) - 2, np.nan))
        value = _geology(x, y) + rng.normal(scale=0.05, size=n_per_line)
        value = value + (offset_nt / 2.0 if reverse else -offset_nt / 2.0)
        xs.append(x)
        ys.append(y)
        comp.append(value)
        prev_end = (x[-1], y[-1])
    x0 = np.concatenate(xs)
    y0 = np.concatenate(ys)
    mag_comp = np.concatenate(comp)
    lat, lon = _xy_to_latlon(x0, y0)
    n = len(lat)
    return pd.DataFrame(
        {
            "Latitude": lat,
            "Longitude": lon,
            "Mag": mag_comp + rng.normal(scale=0.3, size=n),
            "MagComp": mag_comp,
            "CompassX": np.ones(n),
            "CompassY": np.zeros(n),
            "CompassZ": np.zeros(n),
        }
    )


def _write_split(df, tmp_path, n_files=2, prefix="survey"):
    """Split one combined survey into several -comp.csv-shaped files by
    contiguous row ranges, as if it had been flown/recorded as separate
    files -- exercises the CLI's multi-file combine path."""
    edges = np.linspace(0, len(df), n_files + 1).astype(int)
    paths = []
    for i in range(n_files):
        part = df.iloc[edges[i] : edges[i + 1]].reset_index(drop=True)
        path = tmp_path / f"{prefix}{i}-comp.csv"
        write_pre_csv(part, str(path))
        paths.append(str(path))
    return paths


def test_level_survey_recovers_and_removes_injected_offset(tmp_path, capsys):
    df = _survey_df(n_lines=8, offset_nt=6.0)
    paths = _write_split(df, tmp_path, n_files=2)
    out_dir = tmp_path / "out"

    cli.main(["level-survey", "--pre", *paths, "--out-dir", str(out_dir)])
    out = capsys.readouterr().out
    assert "forward/reverse offset" in out
    assert "MagLeveled" in out

    written = sorted(out_dir.glob("*-leveled.csv"))
    assert len(written) == 2

    combined = pd.concat([read_pre_csv(str(p)) for p in written], ignore_index=True)
    assert "MagLeveled" in combined.columns
    # MagLeveled must be built on MagComp (present in every file), not raw Mag.
    before_spread = combined["MagComp"].std()
    after_spread = combined["MagLeveled"].std()
    assert after_spread < before_spread


def test_level_survey_prefers_magcomp_over_mag(tmp_path):
    df = _survey_df(n_lines=8, offset_nt=6.0)
    paths = _write_split(df, tmp_path, n_files=2)
    out_dir = tmp_path / "out"

    cli.main(["level-survey", "--pre", *paths, "--out-dir", str(out_dir)])

    written = sorted(out_dir.glob("*-leveled.csv"))
    combined = pd.concat([read_pre_csv(str(p)) for p in written], ignore_index=True)
    # MagComp carries the injected offset cleanly; Mag has extra noise on
    # top. MagLeveled removes the offset from MagComp, so it should end up
    # tighter than raw Mag (which was never leveled at all).
    assert combined["MagLeveled"].std() < combined["Mag"].std()


def test_level_survey_pre_glob(tmp_path, capsys):
    df = _survey_df(n_lines=8, offset_nt=6.0)
    _write_split(df, tmp_path, n_files=2, prefix="glob_survey")
    out_dir = tmp_path / "out"

    cli.main([
        "level-survey", "--pre-glob", str(tmp_path / "glob_survey*-comp.csv"),
        "--out-dir", str(out_dir),
    ])
    out = capsys.readouterr().out
    assert "leveling 'MagComp' across 2 file(s)" in out
    assert len(list(out_dir.glob("*-leveled.csv"))) == 2


def test_level_survey_declines_with_too_few_lines(tmp_path, capsys):
    df = _survey_df(n_lines=2, offset_nt=6.0)
    paths = _write_split(df, tmp_path, n_files=1)
    out_dir = tmp_path / "out"

    cli.main(["level-survey", "--pre", *paths, "--out-dir", str(out_dir)])
    out = capsys.readouterr().out
    assert "no correction computed" in out
    assert not list(out_dir.glob("*"))


def test_level_survey_still_writes_compensation_when_leveling_is_unreliable(tmp_path, capsys):
    """A --cal compensation that succeeded must not be thrown away just
    because the separate forward/reverse leveling step declines (too few
    lines here) -- found missing when generating real M350 output: the
    command used to write nothing at all in that case."""
    rng = np.random.default_rng(0)
    theta = np.arccos(1 - 2 * (np.arange(200) + 0.5) / 200)
    phi = (2 * np.pi * np.arange(200) / ((1 + 5**0.5) / 2)) % (2 * np.pi)
    vec = np.column_stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    cal_df = pd.DataFrame({
        "Mag": 50000.0 + 2.0 * np.cos(theta) + rng.normal(scale=0.01, size=200),
        "CompassX": vec[:, 0], "CompassY": vec[:, 1], "CompassZ": vec[:, 2],
    })
    from mag_heading.calibration import build_calibration
    model = build_calibration(cal_df, method="nearest")
    cal_path = tmp_path / "test.cal.npz"
    model.save(str(cal_path))

    df = _survey_df(n_lines=2, offset_nt=6.0)
    paths = _write_split(df, tmp_path, n_files=1)
    out_dir = tmp_path / "out"

    cli.main([
        "level-survey", "--pre", *paths, "--cal", str(cal_path), "--out-dir", str(out_dir),
    ])
    out = capsys.readouterr().out
    assert "no correction computed" in out
    assert "MagLeveled NOT added" not in out  # never got far enough to compute one

    written = list(out_dir.glob("*"))
    assert len(written) == 1
    result_df = read_pre_csv(str(written[0]))
    assert "MagComp" in result_df.columns
    assert "MagLeveled" not in result_df.columns


def test_level_survey_no_input_files(capsys):
    cli.main(["level-survey"])
    out = capsys.readouterr().out
    assert "no input files given" in out


def test_level_survey_missing_value_col(tmp_path, capsys):
    df = _survey_df(n_lines=8, offset_nt=6.0)
    paths = _write_split(df, tmp_path, n_files=2)
    out_dir = tmp_path / "out"

    cli.main([
        "level-survey", "--pre", *paths, "--value-col", "DoesNotExist",
        "--out-dir", str(out_dir),
    ])
    out = capsys.readouterr().out
    assert "not found in" in out
    assert not list(out_dir.glob("*"))
