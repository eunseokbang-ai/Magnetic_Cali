import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree

from .heading import compute_heading, unit_vector
from .io import select_segments

VALID_METHODS = ("nearest", "spline")

# A thin-plate-spline RBF carries a degree-1 polynomial term, which in the 3D
# unit-sphere embedding has 4 monomials (1, x, y, z) and therefore needs at
# least 4 distinct centers before the system is solvable. Fewer grid bins than
# this means the segment barely spans any heading at all.
MIN_SPLINE_BINS = 4

CALIBRATION_COLUMNS = ("CompassX", "CompassY", "CompassZ")


def _finite_sample_mask(df_segment, mag_col):
    """Rows whose Mag and all three compass components are finite.

    A single NaN silently poisons everything downstream: np.median returns a NaN
    baseline, every delta_b becomes NaN, and the resulting self-compensation p2p
    is NaN -- which then slips through every `p2p > threshold` rejection gate,
    because comparisons against NaN are always False. Vendor exports carry a
    MagValid flag precisely because dropped/invalid readings do occur, so
    filtering these out is the physically correct step (the Zhang et al. paper
    says the same about dead-zone dropouts), not merely defensive coding.
    """
    columns = [mag_col, *CALIBRATION_COLUMNS]
    values = df_segment[columns].to_numpy(dtype=float)
    return np.isfinite(values).all(axis=1)


DEFAULT_COVERAGE_SAMPLE_LIMIT = 20000


def _coverage_subsample_indices(n, limit, seed=0):
    """Indices of a random survey subsample for coverage estimation, or None for all.

    `survey_uncovered_fraction` is a population mean, so estimating it from a
    random sample is unbiased and its standard error falls as 1/sqrt(n) -- at
    20k samples that is under 0.4 percentage points even in the worst case
    (p=0.5). The KD-tree query over *every* survey sample, repeated once per
    candidate window, is otherwise the dominant cost of a scan: measured on the
    real HaeNam data (27k-sample calibration file, 165k-sample survey) one
    window_size=750 scan took 9.6 s at full resolution versus 1.5 s at 1-in-10,
    with an identical top-5 ranking. The subsample is drawn with a fixed seed so
    repeated runs rank candidates identically, and callers that need the exact
    figure recompute it once for the finally chosen model.
    """
    if limit is None or limit <= 0 or n <= limit:
        return None
    idx = np.random.default_rng(seed).choice(n, size=limit, replace=False)
    idx.sort()
    return idx


def _grid_chord(grid_resolution_deg):
    return 2.0 * np.sin(np.radians(grid_resolution_deg) / 2.0)


def _voxel_bin_keys(unit_vecs, grid_resolution_deg):
    """Integer voxel key per unit vector, for grouping into angular grid cells."""
    return np.floor(unit_vecs / _grid_chord(grid_resolution_deg)).astype(np.int64)


class CalibrationModel:
    """MagArrow heading-error model Delta B(theta, phi), fit from a calibration segment.

    The raw calibration scatter is first binned into angular cells (grid_resolution_deg)
    and averaged per cell before interpolation. This matters for the self-compensation
    quality check: interpolating the raw scatter directly would let each point recover
    itself exactly (zero residual, always), which hides exactly the non-heading noise
    (field gradient, drone interference, diurnal drift) the check is meant to catch.
    Binning first means a raw sample is compared against the average of its neighbors,
    so genuine scatter shows up as nonzero residual -- matching how the vendor's
    LabVIEW tool's "Calibration Data Quality Check" behaves.

    Stores the raw calibration segment (Mag, CompassX/Y/Z) alongside the binned grid,
    so the model is self-contained and can also be re-exported in the vendor `-cal.csv`
    format for comparison/compatibility.
    """

    def __init__(
        self,
        mag_cal,
        compass_x_cal,
        compass_y_cal,
        compass_z_cal,
        method="nearest",
        coverage_threshold_deg=10.0,
        grid_resolution_deg=5.0,
        metadata=None,
    ):
        if method not in VALID_METHODS:
            raise ValueError(f"method must be one of {VALID_METHODS}, got {method!r}")

        self.mag_cal = np.asarray(mag_cal, dtype=float)
        self.compass_x_cal = np.asarray(compass_x_cal, dtype=float)
        self.compass_y_cal = np.asarray(compass_y_cal, dtype=float)
        self.compass_z_cal = np.asarray(compass_z_cal, dtype=float)
        self.method = method
        self.coverage_threshold_deg = float(coverage_threshold_deg)
        self.grid_resolution_deg = float(grid_resolution_deg)
        self.metadata = dict(metadata or {})

        if len(self.mag_cal) == 0:
            raise ValueError("calibration segment is empty; nothing to fit")

        # build_calibration() filters these out, so reaching here with non-finite
        # values means the model was constructed directly (or loaded from a file
        # written before that filtering existed). Fail loudly rather than produce
        # a NaN baseline that quietly defeats every downstream quality gate.
        nonfinite = ~(
            np.isfinite(self.mag_cal)
            & np.isfinite(self.compass_x_cal)
            & np.isfinite(self.compass_y_cal)
            & np.isfinite(self.compass_z_cal)
        )
        if np.any(nonfinite):
            raise ValueError(
                f"{int(nonfinite.sum())} of {len(self.mag_cal)} calibration samples have a "
                "non-finite Mag or compass value; filter them out before building a model "
                "(build_calibration does this automatically)"
            )

        self.baseline = float(np.median(self.mag_cal))
        self.theta_cal, self.phi_cal = compute_heading(
            self.compass_x_cal, self.compass_y_cal, self.compass_z_cal
        )
        self.delta_b_cal = self.mag_cal - self.baseline
        self.unit_vecs_cal = unit_vector(self.theta_cal, self.phi_cal)

        # chord distance = 2*sin(angle/2); linear for small angles, exact bound for large.
        self._coverage_chord = _grid_chord(self.coverage_threshold_deg)

        self.bin_unit_vecs, self.bin_delta_b, self.bin_counts = self._build_grid()

        self._tree = cKDTree(self.bin_unit_vecs)
        self._rbf = None
        if self.method == "spline":
            if len(self.bin_delta_b) < MIN_SPLINE_BINS:
                raise ValueError(
                    f"method='spline' needs at least {MIN_SPLINE_BINS} angular grid bins, but "
                    f"this segment's {len(self.mag_cal)} samples span only "
                    f"{len(self.bin_delta_b)} bin(s) at grid_resolution_deg="
                    f"{self.grid_resolution_deg:g}. The heading barely varies across the "
                    "segment -- use method='nearest', pick a segment covering more headings "
                    "(a turn), or lower grid_resolution_deg."
                )
            self._rbf = RBFInterpolator(
                self.bin_unit_vecs, self.bin_delta_b, kernel="thin_plate_spline"
            )

    def _build_grid(self):
        keys = _voxel_bin_keys(self.unit_vecs_cal, self.grid_resolution_deg)
        grid_df = pd.DataFrame(
            {
                "kx": keys[:, 0],
                "ky": keys[:, 1],
                "kz": keys[:, 2],
                "dB": self.delta_b_cal,
                "ux": self.unit_vecs_cal[:, 0],
                "uy": self.unit_vecs_cal[:, 1],
                "uz": self.unit_vecs_cal[:, 2],
            }
        )
        grouped = grid_df.groupby(["kx", "ky", "kz"], sort=False)
        bin_delta_b = grouped["dB"].mean().to_numpy()
        bin_counts = grouped.size().to_numpy()
        mean_vecs = grouped[["ux", "uy", "uz"]].mean().to_numpy()
        norms = np.linalg.norm(mean_vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        bin_unit_vecs = mean_vecs / norms
        return bin_unit_vecs, bin_delta_b, bin_counts

    def predict(self, theta, phi):
        """Delta B at each (theta, phi). Rows with a non-finite heading yield NaN.

        A survey row whose compass reading dropped out has no usable heading, and
        cKDTree.query raises outright on non-finite input -- so without this such a
        row would abort compensation of the entire file. NaN is the honest answer
        for "no heading, therefore no defensible correction", and it keeps the
        damage confined to that row.
        """
        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        phi = np.atleast_1d(np.asarray(phi, dtype=float))
        query = unit_vector(theta, phi)

        result = np.full(len(query), np.nan)
        usable = np.isfinite(query).all(axis=1)
        if not np.any(usable):
            return result

        nn_dist, nn_idx = self._tree.query(query[usable])
        nn_dist = np.atleast_1d(nn_dist)
        nn_idx = np.atleast_1d(nn_idx)
        usable_result = self.bin_delta_b[nn_idx]

        if self.method == "spline":
            in_coverage = nn_dist <= self._coverage_chord
            if np.any(in_coverage):
                usable_result = usable_result.copy()
                usable_result[in_coverage] = self._rbf(query[usable][in_coverage])

        result[usable] = usable_result
        return result

    def coverage_distance(self, theta, phi):
        """Chord distance from each (theta, phi) query point to the nearest calibration grid cell.

        Non-finite headings return inf, so they count as uncovered rather than
        crashing the KD-tree query -- a row with no heading is certainly not one
        the calibration covers.
        """
        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        phi = np.atleast_1d(np.asarray(phi, dtype=float))
        query = unit_vector(theta, phi)

        result = np.full(len(query), np.inf)
        usable = np.isfinite(query).all(axis=1)
        if np.any(usable):
            nn_dist, _ = self._tree.query(query[usable])
            result[usable] = np.atleast_1d(nn_dist)
        return result

    def save(self, path):
        np.savez(
            path,
            mag_cal=self.mag_cal,
            compass_x_cal=self.compass_x_cal,
            compass_y_cal=self.compass_y_cal,
            compass_z_cal=self.compass_z_cal,
            method=np.array([self.method]),
            coverage_threshold_deg=np.array([self.coverage_threshold_deg]),
            grid_resolution_deg=np.array([self.grid_resolution_deg]),
            metadata_json=np.array([json.dumps(self.metadata)]),
        )

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=False)
        return cls(
            mag_cal=data["mag_cal"],
            compass_x_cal=data["compass_x_cal"],
            compass_y_cal=data["compass_y_cal"],
            compass_z_cal=data["compass_z_cal"],
            method=str(data["method"][0]),
            coverage_threshold_deg=float(data["coverage_threshold_deg"][0]),
            grid_resolution_deg=float(data["grid_resolution_deg"][0])
            if "grid_resolution_deg" in data
            else 5.0,
            metadata=json.loads(str(data["metadata_json"][0])),
        )

    def export_vendor_cal_csv(self, path):
        from .io import write_cal_csv

        df = pd.DataFrame(
            {
                "Mag": self.mag_cal,
                "CompassX": self.compass_x_cal,
                "CompassY": self.compass_y_cal,
                "CompassZ": self.compass_z_cal,
            }
        )
        write_cal_csv(df, path)


def trim_segment_outliers(
    df_segment,
    mag_col="Mag",
    grid_resolution_deg=5.0,
    mad_threshold=3.0,
    min_bin_size_to_trim=3,
    max_trim_fraction_per_bin=0.5,
):
    """Remove non-heading noise spikes from a calibration segment without losing heading coverage.

    More raw samples is not the goal -- the vendor guide's own quality check is
    "field-reading variations can only be from the heading effect", and extra
    points that violate that (field gradient, drone interference) only make the
    self-compensation fit worse. But a point that's the sole (or one of few)
    observations of a given heading must survive even if it looks noisy, since
    discarding it opens a coverage gap right where compensation matters most --
    that's why this trims *within* each angular grid bin (relative to that bin's
    own median), never trims bins smaller than min_bin_size_to_trim, and caps how
    much of any one bin can be removed.
    """
    mag = df_segment[mag_col].to_numpy(dtype=float)
    theta, phi = compute_heading(
        df_segment["CompassX"].to_numpy(),
        df_segment["CompassY"].to_numpy(),
        df_segment["CompassZ"].to_numpy(),
    )
    delta_b = mag - np.median(mag)
    unit_vecs = unit_vector(theta, phi)
    keys = _voxel_bin_keys(unit_vecs, grid_resolution_deg)

    grid_df = pd.DataFrame({"kx": keys[:, 0], "ky": keys[:, 1], "kz": keys[:, 2]})
    keep = np.ones(len(df_segment), dtype=bool)

    for _, positions in grid_df.groupby(["kx", "ky", "kz"], sort=False).indices.items():
        positions = np.asarray(positions)
        if len(positions) < min_bin_size_to_trim:
            continue  # too few points to safely judge -- keep all, coverage matters more

        vals = delta_b[positions]
        med = np.median(vals)
        mad = np.median(np.abs(vals - med))
        if mad == 0:
            continue

        robust_z = 0.6745 * (vals - med) / mad
        outlier_local = np.where(np.abs(robust_z) > mad_threshold)[0]
        if len(outlier_local) == 0:
            continue

        max_removable = min(int(len(positions) * max_trim_fraction_per_bin), len(positions) - 1)
        if len(outlier_local) > max_removable:
            worst_first = np.argsort(-np.abs(robust_z[outlier_local]))
            outlier_local = outlier_local[worst_first[:max_removable]]
        keep[positions[outlier_local]] = False

    trimmed = df_segment[keep].reset_index(drop=True)
    trimmed.attrs.update(df_segment.attrs)
    stats = {
        "n_before": int(len(df_segment)),
        "n_after": int(keep.sum()),
        "n_removed": int((~keep).sum()),
        "fraction_removed": float((~keep).sum() / len(df_segment)) if len(df_segment) else 0.0,
    }
    return trimmed, stats


def build_calibration(
    df_segment,
    mag_col="Mag",
    method="nearest",
    coverage_threshold_deg=10.0,
    grid_resolution_deg=5.0,
    trim_outliers=False,
    outlier_mad_threshold=3.0,
    metadata=None,
):
    meta = dict(metadata or {})
    meta.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    meta["n_samples_before_trim"] = int(len(df_segment))

    finite = _finite_sample_mask(df_segment, mag_col)
    n_nonfinite = int((~finite).sum())
    if n_nonfinite:
        meta["n_nonfinite_dropped"] = n_nonfinite
        if not np.any(finite):
            raise ValueError(
                f"calibration segment has no usable samples: all {len(df_segment)} rows have a "
                f"non-finite {mag_col} or compass value"
            )
        segment_attrs = dict(df_segment.attrs)
        df_segment = df_segment[finite].reset_index(drop=True)
        df_segment.attrs.update(segment_attrs)

    if trim_outliers:
        df_segment, trim_stats = trim_segment_outliers(
            df_segment,
            mag_col=mag_col,
            grid_resolution_deg=grid_resolution_deg,
            mad_threshold=outlier_mad_threshold,
        )
        meta["outlier_trim_stats"] = trim_stats

    meta.setdefault("n_samples", int(len(df_segment)))
    return CalibrationModel(
        mag_cal=df_segment[mag_col].to_numpy(),
        compass_x_cal=df_segment["CompassX"].to_numpy(),
        compass_y_cal=df_segment["CompassY"].to_numpy(),
        compass_z_cal=df_segment["CompassZ"].to_numpy(),
        method=method,
        coverage_threshold_deg=coverage_threshold_deg,
        grid_resolution_deg=grid_resolution_deg,
        metadata=meta,
    )


def quality_report(model, max_variation=None, survey_df=None):
    """Quantify the two calibration-quality criteria from the vendor guide:
    (1) self-compensation flatness, (2) heading coverage vs. the survey data.
    """
    residual = model.mag_cal - model.baseline - model.predict(model.theta_cal, model.phi_cal)
    p2p = float(np.max(residual) - np.min(residual))

    report = {
        "method": model.method,
        "n_calibration_samples": int(len(model.mag_cal)),
        "n_grid_bins": int(len(model.bin_delta_b)),
        "grid_resolution_deg": model.grid_resolution_deg,
        "median_samples_per_bin": float(np.median(model.bin_counts)),
        "bins_with_single_sample_fraction": float(np.mean(model.bin_counts == 1)),
        "self_compensation_p2p_nT": p2p,
        "self_compensation_std_nT": float(np.std(residual)),
    }
    if max_variation is not None:
        report["max_variation_nT"] = float(max_variation)
        report["self_compensation_pass"] = bool(p2p <= max_variation)

    if survey_df is not None:
        theta_s, phi_s = compute_heading(
            survey_df["CompassX"].to_numpy(),
            survey_df["CompassY"].to_numpy(),
            survey_df["CompassZ"].to_numpy(),
        )
        chord = model.coverage_distance(theta_s, phi_s)
        uncovered = chord > model._coverage_chord
        report["survey_heading_coverage_threshold_deg"] = model.coverage_threshold_deg
        report["survey_uncovered_fraction"] = float(np.mean(uncovered))
        report["survey_chord_distance_median"] = float(np.median(chord))
        report["survey_chord_distance_p95"] = float(np.percentile(chord, 95))
        report["survey_chord_distance_max"] = float(np.max(chord))

        dom_theta_deg, dom_phi_deg, dom_fraction = _dominant_heading(
            theta_s, phi_s, model.grid_resolution_deg
        )
        dom_chord = model.coverage_distance(np.radians(dom_theta_deg), np.radians(dom_phi_deg))[0]
        report["survey_dominant_heading_deg"] = (round(dom_theta_deg, 1), round(dom_phi_deg, 1))
        report["survey_dominant_heading_fraction"] = dom_fraction
        report["survey_dominant_heading_covered"] = bool(dom_chord <= model._coverage_chord)

    return report


def _dominant_heading(theta, phi, grid_resolution_deg):
    """The survey's most-sampled heading bin, as a proxy for the drone's primary
    flight-line heading (straight-line flying dominates sample count in most surveys,
    far more than turns). Used to flag whether that specific heading -- the one that
    matters most, since its error repeats systematically across the whole survey --
    is actually covered by the calibration, not just "coverage on average".
    """
    unit_vecs = unit_vector(theta, phi)
    keys = _voxel_bin_keys(unit_vecs, grid_resolution_deg)
    grid_df = pd.DataFrame({"kx": keys[:, 0], "ky": keys[:, 1], "kz": keys[:, 2]})
    counts = grid_df.groupby(["kx", "ky", "kz"], sort=False).size()
    top_key = counts.idxmax()
    mask = (keys[:, 0] == top_key[0]) & (keys[:, 1] == top_key[1]) & (keys[:, 2] == top_key[2])
    dom_theta_deg = float(np.degrees(np.median(theta[mask])))
    dom_phi_deg = float(np.degrees(np.median(phi[mask])))
    dom_fraction = float(np.mean(mask))
    return dom_theta_deg, dom_phi_deg, dom_fraction


def scan_calibration_candidates(
    df,
    window_size,
    step=None,
    method="nearest",
    coverage_threshold_deg=10.0,
    grid_resolution_deg=5.0,
    trim_outliers=False,
    outlier_mad_threshold=3.0,
    max_variation=None,
    top_n=10,
    survey_df=None,
    coverage_sample_limit=DEFAULT_COVERAGE_SAMPLE_LIMIT,
):
    """Slide a fixed-size window over `df` and rank candidate calibration segments.

    The vendor guide's two calibration-quality criteria -- self-compensation
    flatness and heading coverage of the survey -- usually trade off against each
    other (a wider/noisier window covers more headings but is less flat, per the
    guide's own "Module 3" discussion), so both are reported per window rather
    than collapsed into one score. Candidates are ranked by whether they cover the
    survey's dominant heading first (the drone's primary flight-line heading --
    missing it matters more than any other single coverage gap, since that error
    repeats across the whole survey), then by self-compensation p2p, then by
    overall survey heading coverage.

    `survey_df` defaults to `df` itself (the common case: scanning a survey file
    for its own best calibration segment). `survey_uncovered_fraction` is
    estimated from at most `coverage_sample_limit` randomly chosen survey samples
    (pass 0 or None to use every sample) -- see `_coverage_subsample_indices` for
    why that is both safe and a large speedup. The dominant-heading diagnostic is
    always computed from the full survey.
    """
    if window_size < 2:
        raise ValueError("window_size must be >= 2")
    if window_size > len(df):
        raise ValueError(f"window_size ({window_size}) exceeds data length ({len(df)})")

    survey_df = df if survey_df is None else survey_df
    theta_s, phi_s = compute_heading(
        survey_df["CompassX"].to_numpy(),
        survey_df["CompassY"].to_numpy(),
        survey_df["CompassZ"].to_numpy(),
    )
    dom_theta_deg, dom_phi_deg, dom_fraction = _dominant_heading(
        theta_s, phi_s, grid_resolution_deg
    )
    # With too few samples backing it, the "dominant" bin is just noise (e.g. a
    # scattered/uniform heading distribution with no real repeated direction) --
    # don't let an arbitrary single-sample bin drive ranking in that case.
    dom_is_meaningful = (dom_fraction * len(survey_df)) >= 5

    cov_idx = _coverage_subsample_indices(len(theta_s), coverage_sample_limit)
    theta_cov = theta_s if cov_idx is None else theta_s[cov_idx]
    phi_cov = phi_s if cov_idx is None else phi_s[cov_idx]

    step = step or max(1, window_size // 4)
    candidates = []
    for start in range(0, len(df) - window_size + 1, step):
        end = start + window_size - 1
        segment = df.iloc[start : end + 1]
        try:
            model = build_calibration(
                segment,
                method=method,
                coverage_threshold_deg=coverage_threshold_deg,
                grid_resolution_deg=grid_resolution_deg,
                trim_outliers=trim_outliers,
                outlier_mad_threshold=outlier_mad_threshold,
            )
        except ValueError:
            continue  # e.g. a zero compass vector somewhere in this window

        residual = model.mag_cal - model.baseline - model.predict(
            model.theta_cal, model.phi_cal
        )
        p2p = float(np.max(residual) - np.min(residual))
        chord = model.coverage_distance(theta_cov, phi_cov)
        uncovered_fraction = float(np.mean(chord > model._coverage_chord))
        dom_chord = model.coverage_distance(np.radians(dom_theta_deg), np.radians(dom_phi_deg))[0]
        dom_covered = bool(dom_chord <= model._coverage_chord)

        entry = {
            "start": start,
            "end": end,
            "self_compensation_p2p_nT": p2p,
            "survey_uncovered_fraction": uncovered_fraction,
            "dominant_heading_covered": dom_covered,
            "n_grid_bins": int(len(model.bin_delta_b)),
        }
        if trim_outliers:
            entry["n_outliers_removed"] = model.metadata["outlier_trim_stats"]["n_removed"]
        if max_variation is not None:
            entry["self_compensation_pass"] = bool(p2p <= max_variation)
        candidates.append(entry)

    candidates.sort(
        key=lambda c: (
            not c["dominant_heading_covered"] if dom_is_meaningful else False,
            c["self_compensation_p2p_nT"],
            c["survey_uncovered_fraction"],
        )
    )
    return candidates[:top_n]


def _auto_build_passed(report, max_variation, coverage_target, dom_is_meaningful):
    """The three stopping criteria, in NaN-safe form (`not (x <= y)` rejects NaN)."""
    if report is None:
        return False
    return bool(
        report["self_compensation_p2p_nT"] <= max_variation
        and (report["survey_dominant_heading_covered"] or not dom_is_meaningful)
        and report["survey_uncovered_fraction"] <= coverage_target
    )


def _auto_build_at_window_size(
    df,
    window_size,
    step,
    method,
    coverage_threshold_deg,
    grid_resolution_deg,
    max_variation,
    outlier_mad_threshold,
    max_candidates,
    candidate_pool_size,
    max_overlap_fraction,
    coverage_target,
    survey_df,
    dom_is_meaningful,
    metadata,
    coverage_sample_limit,
):
    # candidate_pool_size (how many ranked windows scan hands back) is kept much
    # larger than max_candidates (how many segments we're willing to accept):
    # scan's top-ranked windows tend to cluster as near-duplicate slices of
    # whichever single region of the survey is cleanest, so a pool sized to
    # match max_candidates gets exhausted by that one region alone -- the walk
    # below needs a deep enough pool to actually reach other, complementary
    # parts of the file once the best region's candidates are used up.
    candidates = scan_calibration_candidates(
        df,
        window_size=window_size,
        step=step,
        method=method,
        coverage_threshold_deg=coverage_threshold_deg,
        grid_resolution_deg=grid_resolution_deg,
        trim_outliers=True,
        outlier_mad_threshold=outlier_mad_threshold,
        max_variation=max_variation,
        top_n=candidate_pool_size,
        survey_df=survey_df,
        coverage_sample_limit=coverage_sample_limit,
    )

    def overlaps_too_much(start, end, accepted):
        width = end - start + 1
        for a_start, a_end in accepted:
            overlap = max(0, min(end, a_end) - max(start, a_start) + 1)
            if width and overlap / width > max_overlap_fraction:
                return True
        return False

    accepted_ranges = []
    log = []
    model = None
    report = None

    for c in candidates:
        if len(accepted_ranges) >= max_candidates:
            break
        start, end = c["start"], c["end"]
        if overlaps_too_much(start, end, accepted_ranges):
            log.append({"start": start, "end": end, "accepted": False,
                        "reason": "overlaps an already-accepted segment"})
            continue

        trial_ranges = accepted_ranges + [(start, end)]
        try:
            segment = select_segments(df, trial_ranges, center_segments=True)
            trial_model = build_calibration(
                segment,
                method=method,
                coverage_threshold_deg=coverage_threshold_deg,
                grid_resolution_deg=grid_resolution_deg,
                trim_outliers=True,
                outlier_mad_threshold=outlier_mad_threshold,
                metadata=metadata,
            )
        except ValueError as exc:
            log.append({"start": start, "end": end, "accepted": False, "reason": str(exc)})
            continue

        trial_report = quality_report(trial_model, max_variation=max_variation, survey_df=survey_df)
        # Written as `not (p2p <= limit)` rather than `p2p > limit` so a NaN p2p
        # is rejected instead of slipping through -- every comparison against NaN
        # is False, which would otherwise let a degenerate segment be accepted.
        if not (trial_report["self_compensation_p2p_nT"] <= max_variation):
            log.append({
                "start": start, "end": end, "accepted": False,
                "reason": f"combined p2p {trial_report['self_compensation_p2p_nT']:.3g}nT "
                          f"exceeds max_variation {max_variation:.3g}nT "
                          f"(alone it was {c['self_compensation_p2p_nT']:.3g}nT)",
            })
            continue

        accepted_ranges.append((start, end))
        model, report = trial_model, trial_report
        log.append({
            "start": start, "end": end, "accepted": True,
            "combined_p2p_nT": trial_report["self_compensation_p2p_nT"],
            "combined_uncovered_fraction": trial_report["survey_uncovered_fraction"],
            "combined_dominant_covered": trial_report["survey_dominant_heading_covered"],
        })

        if ((trial_report["survey_dominant_heading_covered"] or not dom_is_meaningful)
                and trial_report["survey_uncovered_fraction"] <= coverage_target):
            break

    passed = _auto_build_passed(report, max_variation, coverage_target, dom_is_meaningful)
    return {
        "model": model,
        "report": report,
        "passed": passed,
        "segments": accepted_ranges,
        "log": log,
        "window_size": window_size,
    }


def _auto_build_rank(result):
    """Lower is better; used to pick the best-effort attempt when nothing passes."""
    if result["model"] is None:
        return (1, float("inf"), float("inf"))
    return (0, result["report"]["self_compensation_p2p_nT"], result["report"]["survey_uncovered_fraction"])


def auto_build_calibration(
    df,
    window_size=(3000, 1500, 750, 300),
    step=None,
    method="nearest",
    coverage_threshold_deg=10.0,
    grid_resolution_deg=5.0,
    max_variation=2.0,
    outlier_mad_threshold=3.0,
    max_candidates=15,
    candidate_pool_size=60,
    max_overlap_fraction=0.3,
    coverage_target=0.1,
    survey_df=None,
    metadata=None,
    coverage_sample_limit=DEFAULT_COVERAGE_SAMPLE_LIMIT,
):
    """Automate the vendor GUI's own "pick a window, check Pass/Fail, try another"
    calibration workflow, using the same quantitative checks this tool already
    computes instead of eyeballing the Calibration Data Quality Check plot.

    `window_size` may be a single int or a sequence of ints, tried widest-first
    (default (3000, 1500, 750, 300), ~5min down to ~30s at 10Hz). How large a
    window can be before real spatial field gradient contaminates it is a
    property of the survey site, not something one fixed number can predict: on
    the Hongcheon reference dataset, 3000-sample windows -- wide enough to
    contain a full turn maneuver's heading sweep, which is where most real
    heading diversity in a boustrophedon survey lives -- failed badly (p2p up to
    68nT) because that survey's real field gradient is steep enough that even a
    single turn's flight distance crosses it, while ~300-sample windows (close
    to the width of the vendor's own hand-picked snippets there, see manual 8.6)
    passed cleanly (p2p ~0.7nT). Trying progressively shorter windows after a
    wide one fails adapts to this automatically instead of making the user
    manually sweep window sizes themselves.

    Algorithm per window size: `scan_calibration_candidates` ranks up to
    `candidate_pool_size` windows by (dominant heading covered, self p2p,
    uncovered fraction) with per-window outlier trimming baked in -- kept much
    larger than `max_candidates` because the top-ranked windows tend to cluster
    as near-duplicate slices of whichever single region of the survey is
    cleanest overall (confirmed on Hongcheon: with the pool sized to match
    max_candidates, all 15 top candidates fell within one ~2500-sample stretch,
    and uncovered_fraction got stuck at ~50% no matter how many of them were
    combined). This walks that pool and greedily *accepts* candidates into a
    running multi-segment pool -- skipping any that overlaps an already-accepted
    one by more than `max_overlap_fraction` (near-duplicates add bulk, not new
    heading coverage) -- but only keeps an addition if the resulting combined,
    baseline-centered pool's self-compensation p2p still passes `max_variation`.
    A candidate that would corrupt the combined fit (the same cross-location
    baseline-offset failure documented in the manual's 8.6 section) is rejected
    and the next-best candidate is tried instead. Stops as soon as the combined
    pool passes AND covers the survey's dominant heading AND its
    uncovered_fraction is at or below `coverage_target`, or once `max_candidates`
    segments have been accepted -- at which point the next window size in the
    sequence is tried. Returns as soon as any size passes.

    Returns a dict with "model" (the final CalibrationModel, or None if nothing
    could be accepted at any size), "report" (its quality_report against
    survey_df, or None), "passed" (bool -- whether the stopping criteria above
    were actually met), "segments" (the accepted (start, end) ranges),
    "window_size" (which size in the sequence produced this result), "log" (its
    per-candidate accept/reject trail), and "attempts" (one summary dict per
    window size tried, in order, each with its own "window_size"/"passed"/"log").
    Callers must not save "model" to disk unless "passed" is True -- an unmet
    stopping criterion means the result is a best-effort, not a validated
    calibration (see manual 9, "self_compensation_pass: False").
    """
    survey_df = df if survey_df is None else survey_df
    theta_s, phi_s = compute_heading(
        survey_df["CompassX"].to_numpy(), survey_df["CompassY"].to_numpy(),
        survey_df["CompassZ"].to_numpy(),
    )
    _, _, dom_fraction = _dominant_heading(theta_s, phi_s, grid_resolution_deg)
    # Same guard as scan_calibration_candidates: with too few samples backing it,
    # the "dominant" bin is arbitrary noise (e.g. a survey with no repeated
    # flight-line heading at all), and requiring it to be covered would make the
    # stopping criterion unsatisfiable forever.
    dom_is_meaningful = (dom_fraction * len(survey_df)) >= 5

    # np.integer is checked explicitly: numpy scalars are not subclasses of int,
    # so a size computed with numpy (e.g. len(df) // 10) would otherwise fall
    # through to list() and raise "not iterable".
    if isinstance(window_size, (int, np.integer)):
        window_sizes = [int(window_size)]
    else:
        window_sizes = [int(w) for w in window_size]

    # The search evaluates coverage against a subsample for speed; only the three
    # compass columns are needed, so this stays cheap even for a large survey.
    cov_idx = _coverage_subsample_indices(len(survey_df), coverage_sample_limit)
    survey_search = (
        survey_df if cov_idx is None
        else survey_df.iloc[cov_idx][list(CALIBRATION_COLUMNS)]
    )

    attempts = []
    best = None
    for ws in window_sizes:
        if ws > len(df):
            continue
        result = _auto_build_at_window_size(
            df, ws, step, method, coverage_threshold_deg, grid_resolution_deg,
            max_variation, outlier_mad_threshold, max_candidates, candidate_pool_size,
            max_overlap_fraction, coverage_target, survey_search, dom_is_meaningful,
            metadata, coverage_sample_limit,
        )
        if result["model"] is not None and survey_search is not survey_df:
            # The accept/stop decisions above used subsampled coverage; the numbers
            # actually reported to the user -- and the pass/fail verdict that gates
            # saving the model -- must be the exact ones.
            result["report"] = quality_report(
                result["model"], max_variation=max_variation, survey_df=survey_df
            )
            result["passed"] = _auto_build_passed(
                result["report"], max_variation, coverage_target, dom_is_meaningful
            )
        attempts.append({"window_size": ws, "passed": result["passed"], "log": result["log"]})
        if result["passed"]:
            result["attempts"] = attempts
            return result
        if best is None or _auto_build_rank(result) < _auto_build_rank(best):
            best = result

    if best is None:
        best = {"model": None, "report": None, "passed": False, "segments": [], "log": [],
                "window_size": None}
    best["attempts"] = attempts
    return best
