"""GPS-track flight-line detection and forward/reverse direction grouping.

This is a companion to compensation.py's per-sample ΔB(theta, phi) heading
correction, not a replacement for it. That correction removes the swing
noise a changing *sensor orientation* injects sample-to-sample; it does
not, by itself, guarantee that two lines flown in opposite directions come
back reading the same absolute level. Real M350 data from the 2026-09
HaeNam survey showed why the two are different problems: even after a
well-validated ΔB(theta, phi) calibration (turn-only noise +7.3%, 42/42
positive), the compensated file-level survey still had a ~5.6 nT
forward/reverse offset between north- and south-bound lines, because the
calibration flight's own attitude range didn't include the drone's real,
much steeper opposite-heading pitch difference on this platform (theta
~63 deg southbound vs ~19 deg northbound near the survey's dominant
azimuth) densely enough to fit it correctly. That residual is a property
of the *whole survey's* forward/reverse lines, not of any single sample,
so it needs the survey's own line geometry to estimate -- which is what
this module provides, for line_leveling.py to use.

Deliberately simple relative to a full production line-detection pipeline
(no turn-buffer trimming, no gap-bridging): it only needs to be good
enough to split a boustrophedon survey into per-pass segments and know
which way each one was flown.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .plot import local_enu


@dataclass
class LineDetectionParams:
    min_line_samples: int = 50
    min_line_length_m: float = 100.0
    # A survey is assumed boustrophedon (lines running back and forth
    # along one dominant compass direction); this is how far a line's net
    # heading may differ from straight-along/straight-back before the
    # split is judged to be a turn rather than a new line.
    angle_tolerance_deg: float = 30.0


def _net_heading_deg(x: np.ndarray, y: np.ndarray) -> float:
    """Direction from a segment's first point to its last, 0-360 deg in
    the atan2(dy, dx) convention -- NOT folded mod 180, so forward and
    reverse passes stay distinguishable."""
    return float(np.degrees(np.arctan2(y[-1] - y[0], x[-1] - x[0])) % 360.0)


def detect_lines(df: pd.DataFrame, params: LineDetectionParams | None = None) -> pd.DataFrame:
    """Split a survey into flight-line passes from its GPS track alone.

    Returns a copy of `df` with added columns `x`, `y` (local-ENU metres,
    one shared reference point for the whole input -- callers combining
    several files must pass them all through in one call, not file by
    file, or the coordinate systems won't agree) and `line_id` (0-based,
    -1 for samples too short/off-azimuth to belong to a clean pass).
    `.attrs["dominant_azimuth_deg"]` holds the survey's primary line
    direction, mod 180 (a line and its reverse share one azimuth here).

    Splits purely on GPS course reversal: a turn is wherever the
    point-to-point track direction swings by more than 90 degrees from
    the recent running heading. Works for the boustrophedon pattern this
    tool is built around; not a general-purpose line editor.
    """
    p = params or LineDetectionParams()
    out = df.reset_index(drop=True).copy()
    x, y = local_enu(out["Latitude"], out["Longitude"])
    out["x"] = x
    out["y"] = y

    dx, dy = np.diff(x), np.diff(y)
    step = np.hypot(dx, dy)
    # A coarse (multi-second) heading estimate is far less noisy than a
    # sample-to-sample one for deciding "is this a turn", since GPS jitter
    # at 10-20 Hz can easily swing the instantaneous bearing by 90+ deg on
    # its own even in dead-straight flight.
    lag = max(1, int(round(len(x) / max(1, (x.max() - x.min()) / 20.0))) if len(x) > 1 else 1)
    lag = min(lag, max(1, len(x) // 4)) or 1
    x_fwd, y_fwd = x[lag:], y[lag:]
    x_bwd, y_bwd = x[:-lag], y[:-lag]
    course = np.degrees(np.arctan2(y_fwd - y_bwd, x_fwd - x_bwd))
    course = np.r_[np.full(lag, np.nan), course[: len(x) - lag]] if len(course) else np.full(len(x), np.nan)
    # align: course[i] describes travel direction around sample i
    course = np.r_[course, np.full(max(0, len(x) - len(course)), np.nan)][: len(x)]

    valid = np.isfinite(course)
    if valid.sum() < 10:
        out["line_id"] = -1
        out.attrs["dominant_azimuth_deg"] = 0.0
        return out

    bins = np.arange(0, 180 + 2, 2.0)
    hist, edges = np.histogram(course[valid] % 180.0, bins=bins)
    dominant_azimuth = float((edges[hist.argmax()] + edges[hist.argmax() + 1]) / 2.0)

    def _circ_diff180(a, b):
        d = (a - b + 90.0) % 180.0 - 90.0
        return np.abs(d)

    on_azimuth = valid & (_circ_diff180(course, dominant_azimuth) <= p.angle_tolerance_deg)

    line_id = np.full(len(x), -1, dtype=int)
    current = -1
    start = None
    next_id = 0
    for i in range(len(x)):
        if not on_azimuth[i]:
            if start is not None:
                _finalize_run(line_id, x, y, start, i, next_id, p)
                next_id += int(line_id[start] >= 0 and line_id[start] == next_id)
                start = None
            continue
        if start is None:
            start = i
    if start is not None:
        _finalize_run(line_id, x, y, start, len(x), next_id, p)

    # Reassign to contiguous 0..N-1 ids (the loop above may leave gaps).
    out["line_id"] = _renumber(line_id)
    out.attrs["dominant_azimuth_deg"] = dominant_azimuth
    return out


def _finalize_run(line_id, x, y, start, stop, next_id, params: LineDetectionParams):
    n = stop - start
    if n < params.min_line_samples:
        return
    length = float(np.hypot(x[stop - 1] - x[start], y[stop - 1] - y[start]))
    if length < params.min_line_length_m:
        return
    line_id[start:stop] = next_id


def _renumber(line_id: np.ndarray) -> np.ndarray:
    out = np.full(len(line_id), -1, dtype=int)
    mapping: dict[int, int] = {}
    for old in line_id:
        if old < 0:
            continue
        if old not in mapping:
            mapping[old] = len(mapping)
    for i, old in enumerate(line_id):
        if old >= 0:
            out[i] = mapping[old]
    return out


def direction_groups(df: pd.DataFrame, dominant_azimuth_deg: float) -> dict[int, str]:
    """"A" for lines flown roughly along `dominant_azimuth_deg`, "B" for
    the reverse -- the forward/reverse split a heading-effect leveling
    correction needs (see line_leveling.py)."""
    groups: dict[int, str] = {}
    for lid in sorted(int(v) for v in df["line_id"].unique() if v >= 0):
        sub = df[df["line_id"] == lid]
        heading = _net_heading_deg(sub["x"].to_numpy(), sub["y"].to_numpy())
        diff = (heading - dominant_azimuth_deg + 180.0) % 360.0 - 180.0
        groups[lid] = "A" if abs(diff) <= 90.0 else "B"
    return groups


def estimate_line_spacing_m(df: pd.DataFrame, dominant_azimuth_deg: float) -> float | None:
    """Median across-track gap between adjacent lines' centre positions."""
    kept = df[df["line_id"] >= 0]
    line_ids = sorted(int(v) for v in kept["line_id"].unique())
    if len(line_ids) < 2:
        return None
    az = np.radians(dominant_azimuth_deg)
    perp = np.array([-np.sin(az), np.cos(az)])
    centres = []
    for lid in line_ids:
        sub = kept[kept["line_id"] == lid]
        centres.append(sub["x"].mean() * perp[0] + sub["y"].mean() * perp[1])
    centres = np.sort(np.array(centres))
    gaps = np.diff(centres)
    gaps = gaps[gaps > 1.0]
    return float(np.median(gaps)) if len(gaps) else None
