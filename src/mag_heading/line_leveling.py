"""Forward/reverse line-leveling: a survey-wide correction for the part of
the heading effect that a per-sample ΔB(theta, phi) calibration can leave
behind.

Why this is a second, separate step after compensation.py's ΔB(theta, phi)
correction, not a tuning of it: that correction is fit from a *calibration
flight* covering many orientations, then applied per sample from the
*survey's* actual orientation at that instant. It is excellent at removing
the fast, sample-to-sample wobble a changing attitude causes (validated at
+28% turn-only noise reduction, 42/42 files positive, on the 2026-09
HaeNam M350 data) but it is only as good, at any one (theta, phi), as how
well the calibration flight's own attitude range and sample density
covered that specific direction. On that same dataset the two opposite
line headings turned out to sit at genuinely different platform attitudes
(theta ~63 deg southbound vs ~19 deg northbound, a bigger split than the
M400 rig's ~47/31 deg) -- a difference the calibration flight (a level
508 m loiter) did not reproduce, so the fitted ΔB at those two specific
points was estimated well enough for the *shape* of the swing noise but
not accurately enough to also null out the ~5.6 nT constant gap between
the two directions. Rebuilding the calibration model several different
ways (denser coverage at that attitude, segments chosen against the real
survey's own coverage, segments drawn from the survey's own turns instead
of the loiter flight) each moved that residual around without removing it
-- see MagArrow-heading-error-calibration's Phase M/N/O notes -- because
the fitted surface's uncertainty at exactly those two points, not the
segments feeding it, was the limit.

The fix is not a better calibration flight; it is a different kind of
measurement. A survey-wide forward/reverse offset is something the
*survey's own opposite-heading lines* can measure directly and far more
robustly than any single calibration flight can, because there are dozens
of independent line pairs to average over instead of one flight's limited
sample of that attitude. This module estimates exactly that -- a single
constant offset between the two flight directions -- from the ΔB(theta,
phi)-compensated survey itself, and removes it, so a downstream GIS/
gridding tool sees an already-levelled -comp.csv rather than having to
re-discover and correct for the same forward/reverse step on its own.

Method ("local_plane"): tile the survey with overlapping neighbourhoods a
couple of line spacings across. Inside each, fit
value = a + b*dx + c*dy + offset*group (group = +-0.5 for the two flight
directions). The plane absorbs whatever real geological gradient runs
through that patch, so the fitted step is what is left over once the
local field is accounted for; how well the model fits is a direct measure
of whether the patch was actually quiet and uniform. The survey-wide
offset is the robust (median, over the neighbourhoods the model explains
best) centre of the per-neighbourhood steps. Requires at least three
distinct lines in a neighbourhood, since with only one line per direction
"the field rises across the gap" and "the reverse line reads high" are the
same number twice.

Adapted from the same method already deployed and measured in the
companion survey-GIS tool (processing/leveling.py, "local_plane" -
93% stripe-ratio reduction on the raw HaeNam M350 block); ported here so
the offset is already gone from -comp.csv before that tool ever sees the
data, rather than depending on a downstream setting being turned on.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .lines import direction_groups

_MIN_LINES_PER_NEIGHBORHOOD = 3
_MIN_POINTS_PER_GROUP = 12


@dataclass
class LineLevelingResult:
    applied: bool
    reason: str | None
    offset_nt: float | None = None
    n_neighborhoods_considered: int = 0
    n_neighborhoods_kept: int = 0
    line_groups: dict = field(default_factory=dict)
    line_shifts: dict = field(default_factory=dict)
    # Spread of the per-neighbourhood offsets that were kept. A real
    # forward/reverse effect is a property of the platform/calibration, so
    # it should read much the same everywhere; a spread comparable to the
    # offset itself means this number is not measuring a real effect and
    # the correction should not be trusted (see `warnings`).
    offset_spread_nt: float | None = None
    residual_nt: float | None = None
    warnings: list = field(default_factory=list)


def compute_line_leveling(
    df: pd.DataFrame,
    value_col: str,
    dominant_azimuth_deg: float,
    line_spacing_m: float | None,
    quiet_percentile: float = 40.0,
    neighborhood_radius_factor: float = 2.0,
) -> LineLevelingResult:
    """Estimate one survey-wide forward/reverse offset for `value_col`.

    `df` must already have `x`, `y`, `line_id` (see lines.detect_lines) and
    should span the *whole* survey being levelled together -- combine all
    the files with one shared local_enu reference before calling this, not
    per file, or lines from different files won't compare correctly.
    """
    kept = df[df["line_id"] >= 0]
    line_ids = sorted(int(v) for v in kept["line_id"].unique())
    if len(line_ids) < _MIN_LINES_PER_NEIGHBORHOOD:
        return LineLevelingResult(
            False,
            f"측선이 {len(line_ids)}개뿐이라 측선간 레벨링을 계산할 수 없습니다 "
            f"(지질 경사와 방향 오차를 분리하려면 최소 {_MIN_LINES_PER_NEIGHBORHOOD}개 필요).",
        )

    groups = direction_groups(kept, dominant_azimuth_deg)
    if len(set(groups.values())) < 2:
        return LineLevelingResult(
            False, "모든 측선이 같은 방향으로 비행되어 측선간 레벨링을 계산할 수 없습니다 (반대 방향 측선 필요).",
            line_groups=groups,
        )

    x = kept["x"].to_numpy(dtype=float)
    y = kept["y"].to_numpy(dtype=float)
    v = kept[value_col].to_numpy(dtype=float)
    lid = kept["line_id"].to_numpy()
    g = np.where(np.vectorize(groups.get)(lid) == "A", 0.5, -0.5)

    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(v)
    x, y, v, lid, g = x[finite], y[finite], v[finite], lid[finite], g[finite]
    if len(x) < 2 * _MIN_POINTS_PER_GROUP:
        return LineLevelingResult(False, "측선간 레벨링을 계산할 유효 자료가 부족합니다.", line_groups=groups)

    spacing = line_spacing_m if line_spacing_m and line_spacing_m > 0 else 50.0
    radius = neighborhood_radius_factor * spacing
    tree = cKDTree(np.column_stack([x, y]))
    gx = np.arange(x.min(), x.max() + spacing, spacing)
    gy = np.arange(y.min(), y.max() + spacing, spacing)
    centres = np.column_stack([m.ravel() for m in np.meshgrid(gx, gy)])

    offsets: list[float] = []
    residuals: list[float] = []
    n_considered = 0
    for cx, cy in centres:
        idx = tree.query_ball_point([cx, cy], radius)
        if len(idx) < 2 * _MIN_POINTS_PER_GROUP:
            continue
        idx = np.asarray(idx)
        gi = g[idx]
        n_a = int(np.sum(gi > 0))
        if n_a < _MIN_POINTS_PER_GROUP or len(idx) - n_a < _MIN_POINTS_PER_GROUP:
            continue
        if np.unique(lid[idx]).size < _MIN_LINES_PER_NEIGHBORHOOD:
            continue
        n_considered += 1

        design = np.column_stack([np.ones(len(idx)), x[idx] - cx, y[idx] - cy, gi])
        coeff, _res, rank, _sv = np.linalg.lstsq(design, v[idx], rcond=None)
        if rank < design.shape[1]:
            continue
        offsets.append(float(coeff[3]))
        residuals.append(float(np.std(v[idx] - design @ coeff)))

    if not offsets:
        return LineLevelingResult(
            False,
            "측선간 레벨링을 추정할 만한 국소 영역을 찾지 못했습니다 - 반대방향 측선이 "
            f"{_MIN_LINES_PER_NEIGHBORHOOD}개 이상 인접한 구간이 필요합니다.",
            n_neighborhoods_considered=n_considered, line_groups=groups,
        )

    offsets_arr = np.array(offsets)
    residual_arr = np.array(residuals)
    threshold = np.percentile(residual_arr, quiet_percentile)
    quiet = residual_arr <= threshold
    if not quiet.any():
        quiet = np.ones_like(residual_arr, dtype=bool)

    offset = float(np.median(offsets_arr[quiet]))
    spread = float(1.4826 * np.median(np.abs(offsets_arr[quiet] - offset)))

    warnings: list[str] = []
    if spread > abs(offset):
        warnings.append(
            f"구역별 오프셋의 산포({spread:.2f}nT)가 추정값({offset:.2f}nT)보다 큽니다 - "
            "방향에 따른 계통 오차가 아니라 측선별 개별 레벨 오차일 가능성이 높습니다. "
            "이 보정을 적용하기 전에 결과를 확인하세요."
        )

    shifts = {lid_: (-offset / 2.0 if groups[lid_] == "A" else offset / 2.0) for lid_ in groups}
    return LineLevelingResult(
        applied=True,
        reason=None,
        offset_nt=offset,
        n_neighborhoods_considered=n_considered,
        n_neighborhoods_kept=int(quiet.sum()),
        line_groups=groups,
        line_shifts=shifts,
        offset_spread_nt=spread,
        residual_nt=float(np.median(residual_arr[quiet])),
        warnings=warnings,
    )


def apply_line_leveling(df: pd.DataFrame, value_col: str, result: LineLevelingResult) -> np.ndarray:
    values = df[value_col].to_numpy(copy=True)
    if not result.applied or not result.line_shifts:
        return values
    shift = df["line_id"].map(result.line_shifts).fillna(0.0).to_numpy()
    return values + shift
