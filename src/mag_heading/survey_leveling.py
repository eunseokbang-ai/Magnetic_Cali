"""Whole-survey forward/reverse leveling over a set of files.

Shared by the `level-survey` CLI subcommand and the GUI's Batch tab. See
line_leveling.py for why this offset exists even after compensation, and
lines.py for how flight lines are detected.
"""
import os

import pandas as pd

from .compensation import apply_compensation, write_comp_csv
from .io import comp_output_path, read_pre_csv
from .line_leveling import apply_line_leveling, compute_line_leveling
from .lines import LineDetectionParams, detect_lines, estimate_line_spacing_m


def leveled_output_path(path):
    path = str(path)
    if path.lower().endswith(".csv"):
        return path[: -len(".csv")] + "-leveled.csv"
    return path + "-leveled.csv"


def level_survey_files(
    paths,
    model=None,
    smooth_hz=None,
    value_col=None,
    out_dir=None,
    force=False,
    line_params=None,
    quiet_percentile=40.0,
    neighborhood_radius_factor=2.0,
    log=print,
):
    """Estimate one survey-wide forward/reverse offset across `paths` together
    and write the results.

    With `model`, each file is compensated first (MagComp) and that compensation
    is always written, even when leveling declines. A MagLeveled column is added
    only when the estimate is computed and reliable (or `force`).

    Returns a summary dict: value_col, n_lines, dominant_azimuth_deg,
    line_spacing_m, result (LineLevelingResult or None), add_leveled,
    written (list of (in_path, out_path)), and status ("ok" / "nothing" /
    "missing_column" / "no_input").
    """
    paths = list(paths)
    summary = {"value_col": None, "n_lines": 0, "dominant_azimuth_deg": None,
               "line_spacing_m": None, "result": None, "add_leveled": False,
               "written": [], "status": "ok"}
    if not paths:
        log("no input files given (use --pre and/or --pre-glob)")
        summary["status"] = "no_input"
        return summary

    pieces = []
    for file_idx, path in enumerate(paths):
        df = read_pre_csv(path)
        if model is not None:
            df = apply_compensation(df, model, keep_raw=True, smooth_hz=smooth_hz)
        df = df.reset_index(drop=True)
        df["_file_idx"] = file_idx
        df["_row_idx"] = df.index
        pieces.append(df)

    if value_col is None:
        value_col = "MagComp" if all("MagComp" in p.columns for p in pieces) else "Mag"
    summary["value_col"] = value_col
    missing = [paths[i] for i, p in enumerate(pieces) if value_col not in p.columns]
    if missing:
        log(f"column '{value_col}' not found in: {', '.join(os.path.basename(m) for m in missing)}")
        summary["status"] = "missing_column"
        return summary
    log(f"leveling '{value_col}' across {len(paths)} file(s)")

    # One combined dataframe, one shared local_enu reference (see lines.py) --
    # detect_lines/estimate_line_spacing_m must see the whole survey at once
    # or coordinates from different files won't agree.
    combined = pd.concat(pieces, ignore_index=True)
    combined = detect_lines(combined, line_params or LineDetectionParams())
    az = combined.attrs["dominant_azimuth_deg"]
    spacing = estimate_line_spacing_m(combined, az)
    n_lines = len(sorted(int(v) for v in combined["line_id"].unique() if v >= 0))
    summary.update(n_lines=n_lines, dominant_azimuth_deg=az, line_spacing_m=spacing)
    spacing_str = f"{spacing:.1f}m" if spacing else "unknown"
    log(f"detected {n_lines} line(s), dominant azimuth={az:.1f} deg, estimated line spacing={spacing_str}")

    result = compute_line_leveling(
        combined, value_col, az, spacing,
        quiet_percentile=quiet_percentile,
        neighborhood_radius_factor=neighborhood_radius_factor,
    )
    summary["result"] = result

    if not result.applied:
        log(f"no correction computed (forward/reverse offset): {result.reason}")
    else:
        log(f"forward/reverse offset: {result.offset_nt:+.3f} nT "
            f"(spread {result.offset_spread_nt:.3f} nT, from "
            f"{result.n_neighborhoods_kept}/{result.n_neighborhoods_considered} neighborhoods)")
        for w in result.warnings:
            log(f"  warning: {w}")

    # Whether to also write a MagLeveled column: only when the estimate is
    # both computed and (reliable or forced). This is independent of
    # whether compensation happened -- an unreliable/undecided leveling
    # estimate should not throw away a perfectly good MagComp compensation
    # that was already computed above.
    add_leveled = result.applied and (not result.warnings or force)
    summary["add_leveled"] = add_leveled
    if result.applied and result.warnings and not force:
        log("MagLeveled NOT added (see warning above -- the estimate is not reliable "
            "enough to trust). Pass --force to add it anyway.")

    if model is None and not add_leveled:
        # Nothing was computed (no calibration) and nothing reliable to level --
        # writing files identical to the input would just be clutter.
        log("nothing to write (no --cal given and no reliable leveling correction)")
        summary["status"] = "nothing"
        return summary

    if add_leveled:
        combined["MagLeveled"] = apply_line_leveling(combined, value_col, result)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    drop_cols = ["_file_idx", "_row_idx", "x", "y", "line_id"]
    for file_idx, path in enumerate(paths):
        piece = combined[combined["_file_idx"] == file_idx].sort_values("_row_idx")
        piece = piece.drop(columns=drop_cols).reset_index(drop=True)
        default_name = comp_output_path(path) if model is not None else leveled_output_path(path)
        out_path = os.path.join(out_dir, os.path.basename(default_name)) if out_dir else default_name
        out_path = write_comp_csv(piece, path, out_path=out_path)
        summary["written"].append((str(path), str(out_path)))
        log(f"  {os.path.basename(path)} -> {out_path}")

    if add_leveled:
        log(f"{len(paths)} file(s) written with a MagLeveled column "
            f"({value_col} + the per-line forward/reverse shift)")
    else:
        log(f"{len(paths)} file(s) written with {value_col} (no MagLeveled column added)")
    return summary
