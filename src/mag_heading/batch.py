import os

import numpy as np

from .compensation import apply_compensation, write_comp_csv
from .io import comp_output_path, read_pre_csv


def compute_noise_metrics(values):
    """Noise metrics for a Mag column.

    motion_noise_std_nT (the sample-to-sample first-difference std) is the proxy
    for the "swing"-induced fluctuation this whole calibration exists to reduce --
    unlike raw std/p2p, it isn't inflated by genuine slow-varying field anomalies
    the survey is trying to detect, only by sample-to-sample jitter.
    """
    values = np.asarray(values, dtype=float)
    return {
        "std_nT": float(np.std(values)),
        "p2p_nT": float(np.max(values) - np.min(values)),
        "motion_noise_std_nT": float(np.std(np.diff(values))),
    }


def _masked_motion_noise(values, mask):
    """First-difference std over consecutive sample pairs that are both in `mask`
    (so a difference never straddles a turn/line boundary)."""
    values = np.asarray(values, dtype=float)
    pair = mask[:-1] & mask[1:]
    if pair.sum() < 10:
        return None
    d = np.diff(values)[pair]
    d = d[np.isfinite(d)]
    return float(np.std(d)) if len(d) >= 10 else None


def _improvement_pct(raw, comp):
    if raw is None or comp is None or raw <= 0:
        return None
    return 100.0 * (1.0 - comp / raw)


TURN_COURSE_LAG_M = 20.0


def turn_line_masks(df):
    """Boolean (turn_mask, line_mask) from the GPS track, or None when the file
    has no usable Latitude/Longitude.

    Samples on a detected straight flight-line pass are "line"; everything else
    (turns, loiters, short off-azimuth legs) is "turn". Turns are where heading
    error swings hardest, so they are where a calibration has to show it actually
    works -- a calibration can look fine on overall noise while doing nothing in
    the turns (2026-08 HaeNam lesson). Straight lines are where it must not add
    noise (the reason smooth_hz exists).
    """
    if "Latitude" not in df.columns or "Longitude" not in df.columns:
        return None
    from .lines import LineDetectionParams, detect_lines  # local import: lines imports plot

    try:
        # same path-length course lag as level-survey's line detection
        # (see LineDetectionParams.course_lag_m)
        lined = detect_lines(df, LineDetectionParams(course_lag_m=TURN_COURSE_LAG_M))
    except Exception:
        return None
    line_id = lined["line_id"].to_numpy()
    line_mask = line_id >= 0
    return ~line_mask, line_mask


def noise_breakdown(raw_values, comp_values, masks=None):
    """Motion-noise before/after for the whole file, turns only, and lines only.

    `masks` is the (turn_mask, line_mask) pair from turn_line_masks(); the
    turn/line entries are None when it is None or a part has too few samples.
    """
    raw_all = compute_noise_metrics(raw_values)["motion_noise_std_nT"]
    comp_all = compute_noise_metrics(comp_values)["motion_noise_std_nT"]
    out = {
        "raw_motion_noise_std_nT": raw_all,
        "comp_motion_noise_std_nT": comp_all,
        "motion_noise_improvement_pct": _improvement_pct(raw_all, comp_all) or 0.0,
        "turn_fraction": None,
        "raw_turn_noise_std_nT": None,
        "comp_turn_noise_std_nT": None,
        "turn_improvement_pct": None,
        "raw_line_noise_std_nT": None,
        "comp_line_noise_std_nT": None,
        "line_improvement_pct": None,
    }
    if masks is None:
        return out
    turn_mask, line_mask = masks
    out["turn_fraction"] = float(np.mean(turn_mask))
    for key, mask in (("turn", turn_mask), ("line", line_mask)):
        raw = _masked_motion_noise(raw_values, mask)
        comp = _masked_motion_noise(comp_values, mask)
        out[f"raw_{key}_noise_std_nT"] = raw
        out[f"comp_{key}_noise_std_nT"] = comp
        out[f"{key}_improvement_pct"] = _improvement_pct(raw, comp)
    return out


def batch_compensate(pre_paths, model, mag_col="Mag", keep_raw=True, out_dir=None,
                      smooth_hz=None, progress=None, stop_on_error=False):
    """Apply one calibration model to many -pre.csv files and report before/after
    noise reduction per file, so a calibration's real-world effect can be checked
    across an entire field survey before committing to it (rather than trusting a
    single candidate segment's self-compensation score alone).

    A file that fails (unreadable, missing columns, ...) gets a result row with an
    "error" key instead of aborting the whole batch, unless stop_on_error is set.
    `progress(i, n, path)` is called before each file.
    """
    comp_col = "MagComp" if keep_raw else mag_col
    results = []
    pre_paths = list(pre_paths)

    for i, path in enumerate(pre_paths):
        if progress is not None:
            progress(i, len(pre_paths), path)
        try:
            df = read_pre_csv(path)
            compensated = apply_compensation(
                df, model, mag_col=mag_col, keep_raw=keep_raw, smooth_hz=smooth_hz
            )

            if out_dir is not None:
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, os.path.basename(comp_output_path(path)))
            else:
                out_path = None
            out_path = write_comp_csv(compensated, path, out_path=out_path)

            raw_values = df[mag_col].to_numpy()
            comp_values = compensated[comp_col].to_numpy()
            raw_metrics = compute_noise_metrics(raw_values)
            comp_metrics = compute_noise_metrics(comp_values)
            row = {
                "path": str(path),
                "out_path": str(out_path),
                "n_samples": int(len(df)),
                "raw_p2p_nT": raw_metrics["p2p_nT"],
                "comp_p2p_nT": comp_metrics["p2p_nT"],
            }
            row.update(noise_breakdown(raw_values, comp_values, turn_line_masks(df)))
            results.append(row)
        except Exception as exc:
            if stop_on_error:
                raise
            results.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})

    return results
