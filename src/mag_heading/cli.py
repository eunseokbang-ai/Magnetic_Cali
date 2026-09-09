import argparse
import glob
import os
import sys

import pandas as pd

from .batch import batch_compensate
from .calibration import (
    CalibrationModel,
    auto_build_calibration,
    build_calibration,
    quality_report,
    scan_calibration_candidates,
)
from .compensation import apply_compensation, write_comp_csv
from .io import comp_output_path, read_pre_csv, select_segments
from .line_leveling import apply_line_leveling, compute_line_leveling
from .lines import LineDetectionParams, detect_lines, estimate_line_spacing_m
from .plot import (
    plot_2d_comparison,
    plot_calibration_quality,
    plot_compensation_timeseries,
    plot_heading_timeseries,
)


def _cmd_calibrate(args):
    if len(args.start) != len(args.end):
        raise ValueError(f"--start ({len(args.start)} values) and --end ({len(args.end)} values) "
                          "must be given the same number of times")
    df = read_pre_csv(args.pre)
    ranges = list(zip(args.start, args.end))
    segment = select_segments(df, ranges, center_segments=not args.no_center_segments)
    model = build_calibration(
        segment,
        method=args.method,
        coverage_threshold_deg=args.coverage_threshold_deg,
        grid_resolution_deg=args.grid_resolution_deg,
        trim_outliers=args.trim_outliers,
        outlier_mad_threshold=args.outlier_mad_threshold,
        metadata={
            "source_pre_csv": str(args.pre),
            "segments": [f"{s}-{e}" for s, e in ranges],
        },
    )
    model.save(args.out)
    n_used = model.metadata["n_samples"]
    print(f"calibration model saved to {args.out} ({n_used} samples, method={args.method})")
    offsets = segment.attrs.get("segment_baseline_offsets_nT")
    if offsets:
        offsets_str = ", ".join(f"{o:+.2f}" for o in offsets)
        print(f"  segment baseline offsets applied (nT): {offsets_str}")
        print("  (segments were picked at different points in the survey, so each was "
              "re-centered to a common field level before pooling -- see manual 8.6)")
    if args.trim_outliers:
        stats = model.metadata["outlier_trim_stats"]
        print(f"  outlier trim: removed {stats['n_removed']}/{stats['n_before']} samples "
              f"({stats['fraction_removed']:.1%})")

    if args.export_vendor_cal:
        model.export_vendor_cal_csv(args.export_vendor_cal)
        print(f"vendor-format -cal.csv exported to {args.export_vendor_cal}")

    report = quality_report(model, max_variation=args.max_variation)
    _print_report(report)


def _cmd_check(args):
    model = CalibrationModel.load(args.cal)
    survey_df = read_pre_csv(args.pre) if args.pre else None
    report = quality_report(model, max_variation=args.max_variation, survey_df=survey_df)
    _print_report(report)


def _cmd_compensate(args):
    model = CalibrationModel.load(args.cal)
    df = read_pre_csv(args.pre)
    compensated = apply_compensation(
        df, model, keep_raw=not args.overwrite, smooth_hz=args.smooth_hz
    )
    out_path = write_comp_csv(compensated, args.pre, out_path=args.out)
    print(f"compensated survey written to {out_path} ({len(compensated)} samples)")


def _cmd_batch_compensate(args):
    model = CalibrationModel.load(args.cal)

    paths = list(args.pre or [])
    if args.pre_glob:
        paths.extend(sorted(glob.glob(args.pre_glob)))
    if not paths:
        print("no input files given (use --pre and/or --pre-glob)")
        return

    results = batch_compensate(
        paths, model, keep_raw=not args.overwrite, out_dir=args.out_dir,
        smooth_hz=args.smooth_hz,
    )

    header = f"{'file':<40} {'n':>7} {'raw_noise':>10} {'comp_noise':>11} {'improve':>9}"
    print(header)
    for r in results:
        name = os.path.basename(r["path"])
        print(
            f"{name:<40.40} {r['n_samples']:>7} {r['raw_motion_noise_std_nT']:>10.4g} "
            f"{r['comp_motion_noise_std_nT']:>11.4g} {r['motion_noise_improvement_pct']:>8.1f}%"
        )

    improvements = [r["motion_noise_improvement_pct"] for r in results]
    print()
    print(f"mean motion-noise improvement: {sum(improvements) / len(improvements):.1f}%  "
          f"(min {min(improvements):.1f}%, max {max(improvements):.1f}%)")
    poor = [os.path.basename(r["path"]) for r in results if r["motion_noise_improvement_pct"] < 5.0]
    if poor:
        print(f"files with <5% improvement (consider rebuilding the calibration): {', '.join(poor)}")


def _leveled_output_path(path):
    path = str(path)
    if path.lower().endswith(".csv"):
        return path[: -len(".csv")] + "-leveled.csv"
    return path + "-leveled.csv"


def _cmd_level_survey(args):
    """Estimate and (if reliable) remove one survey-wide forward/reverse
    offset from a whole set of -pre.csv/-comp.csv files together.

    This is deliberately a separate step from `compensate`/`batch-compensate`:
    the per-sample ΔB(theta, phi) correction only knows about one file's own
    orientation history, while the offset this command measures is a property
    of the *whole survey's* opposite-heading lines and needs all of them
    combined, with one shared local coordinate reference, to estimate
    robustly. See line_leveling.py's module docstring for why this is needed
    even after compensation, and lines.py's for the line-detection method.
    """
    paths = list(args.pre or [])
    if args.pre_glob:
        paths.extend(sorted(glob.glob(args.pre_glob)))
    if not paths:
        print("no input files given (use --pre and/or --pre-glob)")
        return

    model = CalibrationModel.load(args.cal) if args.cal else None

    pieces = []
    for file_idx, path in enumerate(paths):
        df = read_pre_csv(path)
        if model is not None:
            df = apply_compensation(df, model, keep_raw=True, smooth_hz=args.smooth_hz)
        df = df.reset_index(drop=True)
        df["_file_idx"] = file_idx
        df["_row_idx"] = df.index
        pieces.append(df)

    value_col = args.value_col
    if value_col is None:
        value_col = "MagComp" if all("MagComp" in p.columns for p in pieces) else "Mag"
    missing = [paths[i] for i, p in enumerate(pieces) if value_col not in p.columns]
    if missing:
        print(f"column '{value_col}' not found in: {', '.join(os.path.basename(m) for m in missing)}")
        return
    print(f"leveling '{value_col}' across {len(paths)} file(s)")

    # One combined dataframe, one shared local_enu reference (see lines.py) --
    # detect_lines/estimate_line_spacing_m must see the whole survey at once
    # or coordinates from different files won't agree.
    combined = pd.concat(pieces, ignore_index=True)
    combined = detect_lines(
        combined,
        LineDetectionParams(
            min_line_samples=args.min_line_samples,
            min_line_length_m=args.min_line_length_m,
            angle_tolerance_deg=args.angle_tolerance_deg,
        ),
    )
    az = combined.attrs["dominant_azimuth_deg"]
    spacing = estimate_line_spacing_m(combined, az)
    n_lines = len(sorted(int(v) for v in combined["line_id"].unique() if v >= 0))
    spacing_str = f"{spacing:.1f}m" if spacing else "unknown"
    print(f"detected {n_lines} line(s), dominant azimuth={az:.1f} deg, estimated line spacing={spacing_str}")

    result = compute_line_leveling(
        combined, value_col, az, spacing,
        quiet_percentile=args.quiet_percentile,
        neighborhood_radius_factor=args.neighborhood_radius_factor,
    )

    if not result.applied:
        print(f"no correction computed (forward/reverse offset): {result.reason}")
    else:
        print(f"forward/reverse offset: {result.offset_nt:+.3f} nT "
              f"(spread {result.offset_spread_nt:.3f} nT, from "
              f"{result.n_neighborhoods_kept}/{result.n_neighborhoods_considered} neighborhoods)")
        for w in result.warnings:
            print(f"  warning: {w}")

    # Whether to also write a MagLeveled column: only when the estimate is
    # both computed and (reliable or forced). This is independent of
    # whether --cal compensation happened -- an unreliable/undecided
    # leveling estimate should not throw away a perfectly good MagComp
    # compensation that was already computed above.
    add_leveled = result.applied and (not result.warnings or args.force)
    if result.applied and result.warnings and not args.force:
        print("MagLeveled NOT added (see warning above -- the estimate is not reliable "
              "enough to trust). Pass --force to add it anyway.")

    if model is None and not add_leveled:
        # Nothing was computed (no --cal) and nothing reliable to level --
        # writing files identical to the input would just be clutter.
        print("nothing to write (no --cal given and no reliable leveling correction)")
        return

    if add_leveled:
        combined["MagLeveled"] = apply_line_leveling(combined, value_col, result)

    out_dir = args.out_dir
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    drop_cols = ["_file_idx", "_row_idx", "x", "y", "line_id"]
    for file_idx, path in enumerate(paths):
        piece = combined[combined["_file_idx"] == file_idx].sort_values("_row_idx")
        piece = piece.drop(columns=drop_cols).reset_index(drop=True)
        default_name = comp_output_path(path) if model is not None else _leveled_output_path(path)
        out_path = os.path.join(out_dir, os.path.basename(default_name)) if out_dir else default_name
        out_path = write_comp_csv(piece, path, out_path=out_path)
        print(f"  {os.path.basename(path)} -> {out_path}")

    if add_leveled:
        print(f"{len(paths)} file(s) written with a MagLeveled column "
              f"({value_col} + the per-line forward/reverse shift)")
    else:
        print(f"{len(paths)} file(s) written with {value_col} (no MagLeveled column added)")


def _cmd_scan(args):
    df = read_pre_csv(args.pre)
    if args.window_size > len(df):
        # --window-size has a default, so a short file would otherwise fail deep
        # inside scan_calibration_candidates with a bare traceback.
        suggested = max(2, len(df) // 4)
        print(f"--window-size {args.window_size} exceeds this file's {len(df)} samples.")
        print(f"Pass a smaller window, e.g. --window-size {suggested}.")
        return
    candidates = scan_calibration_candidates(
        df,
        window_size=args.window_size,
        step=args.step,
        coverage_sample_limit=args.coverage_sample_limit,
        method=args.method,
        coverage_threshold_deg=args.coverage_threshold_deg,
        grid_resolution_deg=args.grid_resolution_deg,
        trim_outliers=args.trim_outliers,
        outlier_mad_threshold=args.outlier_mad_threshold,
        max_variation=args.max_variation,
        top_n=args.top_n,
    )
    if not candidates:
        print("no candidate windows found")
        return

    header = (f"{'start':>8} {'end':>8} {'p2p_nT':>10} {'uncovered':>10} "
              f"{'dom_cov':>8} {'grid_bins':>10}")
    if args.trim_outliers:
        header += f" {'trimmed':>8}"
    if args.max_variation is not None:
        header += f" {'pass':>6}"
    print(header)
    for c in candidates:
        line = (
            f"{c['start']:>8} {c['end']:>8} {c['self_compensation_p2p_nT']:>10.4g} "
            f"{c['survey_uncovered_fraction']:>10.4g} {str(c['dominant_heading_covered']):>8} "
            f"{c['n_grid_bins']:>10}"
        )
        if args.trim_outliers:
            line += f" {c['n_outliers_removed']:>8}"
        if args.max_variation is not None:
            line += f" {str(c['self_compensation_pass']):>6}"
        print(line)
    print()
    print("dom_cov = does this window cover the survey's dominant (primary flight-line) heading")
    print("use the best candidate's start/end with `mag_heading calibrate --start <start> --end <end>`")


def _cmd_auto_calibrate(args):
    df = read_pre_csv(args.pre)
    result = auto_build_calibration(
        df,
        window_size=args.window_size,
        step=args.step,
        method=args.method,
        coverage_threshold_deg=args.coverage_threshold_deg,
        grid_resolution_deg=args.grid_resolution_deg,
        max_variation=args.max_variation,
        outlier_mad_threshold=args.outlier_mad_threshold,
        max_candidates=args.max_candidates,
        candidate_pool_size=args.candidate_pool_size,
        max_overlap_fraction=args.max_overlap_fraction,
        coverage_target=args.coverage_target,
        coverage_sample_limit=args.coverage_sample_limit,
        metadata={"source_pre_csv": str(args.pre)},
    )

    for attempt in result["attempts"]:
        status = "PASSED" if attempt["passed"] else "did not pass"
        print(f"--- window_size={attempt['window_size']} ({status}) ---")
        for entry in attempt["log"]:
            if entry["accepted"]:
                print(f"  {entry['start']:>8}-{entry['end']:<8} accepted "
                      f"(combined p2p={entry['combined_p2p_nT']:.3g}nT, "
                      f"uncovered={entry['combined_uncovered_fraction']:.1%}, "
                      f"dom_cov={entry['combined_dominant_covered']})")
            else:
                print(f"  {entry['start']:>8}-{entry['end']:<8} rejected ({entry['reason']})")
    print()

    if not result["segments"] or result["model"] is None:
        print("no calibration could be assembled at any --window-size tried -- no candidate "
              "window ever satisfied --max-variation, even alone. Try smaller window sizes, "
              "loosen --max-variation, or pick segments manually with `calibrate`/the GUI.")
        return

    print(f"best result: window_size={result['window_size']}, "
          f"{len(result['segments'])} segment(s): "
          + ", ".join(f"{s}-{e}" for s, e in result["segments"]))
    _print_report(result["report"])

    if result["passed"]:
        result["model"].save(args.out)
        print(f"PASSED -- calibration model saved to {args.out}")
        if args.export_vendor_cal:
            result["model"].export_vendor_cal_csv(args.export_vendor_cal)
            print(f"vendor-format -cal.csv exported to {args.export_vendor_cal}")
    else:
        print("NOT PASSED -- stopping criteria (p2p / dominant heading coverage / coverage "
              "target) were not all met within --max-candidates windows considered.")
        if args.force_save:
            result["model"].save(args.out)
            print(f"--force-save given: saved the best-effort (non-passing) model to {args.out} "
                  "anyway -- review the quality report above before trusting it.")
        else:
            print(f"not saving to {args.out} -- rerun with a larger --max-candidates or looser "
                  "--coverage-target/--max-variation, or use `calibrate`/the GUI to pick segments "
                  "manually. Pass --force-save to save the best-effort model anyway.")


def _cmd_plot2d(args):
    df = read_pre_csv(args.pre)
    comp_col = args.comp_col if args.comp_col in df.columns else None
    if args.comp_col and comp_col is None:
        print(f"warning: column '{args.comp_col}' not found, plotting raw only")
    fig = plot_2d_comparison(df, raw_col=args.raw_col, comp_col=comp_col, resolution_m=args.resolution_m)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"2D plot saved to {args.out}")


def _cmd_plot_heading(args):
    df = read_pre_csv(args.pre)
    fig = plot_heading_timeseries(df)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"heading plot saved to {args.out}")


def _cmd_plot_calibration(args):
    model = CalibrationModel.load(args.cal)
    fig = plot_calibration_quality(model)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"calibration quality plot saved to {args.out}")


def _cmd_plot_timeseries(args):
    df = read_pre_csv(args.pre)
    fig = plot_compensation_timeseries(df, raw_col=args.raw_col, comp_col=args.comp_col)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"compensation time series plot saved to {args.out}")


def _cmd_gui(args):
    from .gui import main as gui_main

    gui_main(args.pre)


def _print_report(report):
    for key, value in report.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4g}")
        else:
            print(f"  {key}: {value}")


def build_parser():
    parser = argparse.ArgumentParser(prog="mag_heading", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_cal = sub.add_parser("calibrate", help="build a calibration model from a -pre.csv segment")
    p_cal.add_argument("--pre", required=True, help="path to a -pre.csv survey file")
    p_cal.add_argument("--start", required=True, nargs="+",
                        help="row index or HH:MM:SS time. Give multiple values (paired "
                             "positionally with --end) to combine several short segments into "
                             "one calibration -- often better than one long window, see manual 8.6")
    p_cal.add_argument("--end", required=True, nargs="+",
                        help="row index or HH:MM:SS time, one per --start value")
    p_cal.add_argument("--no-center-segments", action="store_true",
                        help="when combining multiple --start/--end segments, skip re-centering "
                             "each one to a common field level before pooling. Off by default: "
                             "segments picked from different points in a survey are usually at "
                             "different real field levels, and without this a single global "
                             "baseline misattributes that spatial spread to heading, corrupting "
                             "the fit (see manual 8.6). Has no effect with a single segment.")
    p_cal.add_argument("--method", choices=["nearest", "spline"], default="nearest")
    p_cal.add_argument("--coverage-threshold-deg", type=float, default=10.0)
    p_cal.add_argument("--grid-resolution-deg", type=float, default=5.0)
    p_cal.add_argument("--trim-outliers", action="store_true",
                        help="remove non-heading noise spikes per angular bin before fitting "
                             "(never empties a bin, so heading coverage is preserved)")
    p_cal.add_argument("--outlier-mad-threshold", type=float, default=3.0,
                        help="robust z-score threshold for --trim-outliers")
    p_cal.add_argument("--out", required=True, help="output calibration model path (.npz)")
    p_cal.add_argument("--export-vendor-cal", help="also write a vendor-format -cal.csv")
    p_cal.add_argument("--max-variation", type=float, default=None,
                        help="nT peak-to-peak threshold for the self-compensation pass/fail check")
    p_cal.set_defaults(func=_cmd_calibrate)

    p_check = sub.add_parser("check", help="report calibration quality metrics")
    p_check.add_argument("--cal", required=True, help="path to a calibration model (.npz)")
    p_check.add_argument("--pre", help="optional -pre.csv survey file to check heading coverage against")
    p_check.add_argument("--max-variation", type=float, default=None)
    p_check.set_defaults(func=_cmd_check)

    p_comp = sub.add_parser("compensate", help="apply a calibration model to a -pre.csv survey file")
    p_comp.add_argument("--pre", required=True, help="path to a -pre.csv survey file")
    p_comp.add_argument("--cal", required=True, help="path to a calibration model (.npz)")
    p_comp.add_argument("--out", default=None, help="output path (default: <pre>-comp.csv)")
    p_comp.add_argument("--overwrite", action="store_true",
                         help="overwrite the Mag column instead of adding a MagComp column")
    p_comp.add_argument("--smooth-hz", type=float, default=None,
                         help="low-pass the heading correction at this frequency (Hz) before subtracting it. The correction is computed per sample from the compass and so carries the compass's broadband noise, while the heading effect itself varies well under 1 Hz -- without this, compensation can add more noise than it removes on straight survey lines (measured: up to -27%% before, +6%% after). Try 1.0 for 10 Hz data; 0.2 over-smooths. Off by default (see manual 8.8)")
    p_comp.set_defaults(func=_cmd_compensate)

    p_batch = sub.add_parser(
        "batch-compensate",
        help="apply one calibration model to many -pre.csv files and report per-file "
             "motion-noise reduction",
    )
    p_batch.add_argument("--pre", nargs="+", default=None, help="one or more -pre.csv paths")
    p_batch.add_argument("--pre-glob", default=None,
                          help="glob pattern for -pre.csv files, e.g. 'D:/survey/*-pre.csv'")
    p_batch.add_argument("--cal", required=True, help="path to a calibration model (.npz)")
    p_batch.add_argument("--out-dir", default=None,
                          help="directory for -comp.csv outputs (default: next to each input file)")
    p_batch.add_argument("--overwrite", action="store_true",
                          help="overwrite the Mag column instead of adding a MagComp column")
    p_batch.add_argument("--smooth-hz", type=float, default=None,
                         help="low-pass the heading correction at this frequency (Hz) before subtracting it. The correction is computed per sample from the compass and so carries the compass's broadband noise, while the heading effect itself varies well under 1 Hz -- without this, compensation can add more noise than it removes on straight survey lines (measured: up to -27%% before, +6%% after). Try 1.0 for 10 Hz data; 0.2 over-smooths. Off by default (see manual 8.8)")
    p_batch.set_defaults(func=_cmd_batch_compensate)

    p_level = sub.add_parser(
        "level-survey",
        help="estimate and remove one survey-wide forward/reverse (heading-direction) offset "
             "across a whole set of -pre.csv/-comp.csv files, combined",
    )
    p_level.add_argument("--pre", nargs="+", default=None, help="one or more -pre.csv or -comp.csv paths")
    p_level.add_argument("--pre-glob", default=None,
                          help="glob pattern for input files, e.g. 'D:/survey/*-comp.csv'")
    p_level.add_argument("--cal", default=None,
                          help="optional calibration model (.npz) -- if given, each file is first "
                               "compensated with it (like `batch-compensate`) before leveling; if "
                               "omitted, --pre/--pre-glob are assumed to already carry the column "
                               "to level (e.g. existing -comp.csv files with a MagComp column)")
    p_level.add_argument("--smooth-hz", type=float, default=None,
                          help="passed through to compensation when --cal is given, see `compensate --smooth-hz`")
    p_level.add_argument("--value-col", default=None,
                          help="column to level (default: MagComp if every file has it, else Mag)")
    p_level.add_argument("--out-dir", default=None,
                          help="directory for outputs (default: next to each input file)")
    p_level.add_argument("--min-line-samples", type=int, default=50)
    p_level.add_argument("--min-line-length-m", type=float, default=100.0)
    p_level.add_argument("--angle-tolerance-deg", type=float, default=30.0)
    p_level.add_argument("--quiet-percentile", type=float, default=40.0,
                          help="use only the this-percentile-lowest-residual neighborhoods when "
                               "taking the robust median offset (default 40)")
    p_level.add_argument("--neighborhood-radius-factor", type=float, default=2.0,
                          help="neighborhood radius = this * estimated line spacing (default 2.0)")
    p_level.add_argument("--force", action="store_true",
                          help="write the correction even when it is flagged unreliable (spread "
                               ">= offset) -- review the printed warning first")
    p_level.set_defaults(func=_cmd_level_survey)

    p_scan = sub.add_parser(
        "scan", help="slide a fixed-size window over a -pre.csv and rank candidate calibration segments"
    )
    p_scan.add_argument("--pre", required=True, help="path to a -pre.csv survey file")
    p_scan.add_argument("--window-size", type=int, default=3000,
                         help="candidate window size in samples (default 3000, ~5min at 10Hz -- wide "
                              "enough to contain a full turn maneuver's heading sweep, where most real "
                              "heading diversity in a boustrophedon survey lives; a short window like "
                              "the old 500-sample default usually can't capture that, see manual 8.7)")
    p_scan.add_argument("--step", type=int, default=None, help="stride between windows (default: window-size // 4)")
    p_scan.add_argument("--method", choices=["nearest", "spline"], default="nearest")
    p_scan.add_argument("--coverage-threshold-deg", type=float, default=10.0)
    p_scan.add_argument("--grid-resolution-deg", type=float, default=5.0)
    p_scan.add_argument("--trim-outliers", action="store_true")
    p_scan.add_argument("--outlier-mad-threshold", type=float, default=3.0)
    p_scan.add_argument("--max-variation", type=float, default=None)
    p_scan.add_argument("--top-n", type=int, default=10)
    p_scan.add_argument("--coverage-sample-limit", type=int, default=20000,
                         help="estimate heading coverage from at most this many randomly "
                              "chosen survey samples (0 = use every sample). The per-window "
                              "coverage query over the whole survey is the dominant cost of a "
                              "scan; sampling makes it several times faster with a ranking "
                              "that matches the exact computation")
    p_scan.set_defaults(func=_cmd_scan)

    p_auto = sub.add_parser(
        "auto-calibrate",
        help="automatically assemble and validate a multi-segment calibration from a -pre.csv, "
             "saving it only if it actually passes (see manual 8.7)",
    )
    p_auto.add_argument("--pre", required=True, help="path to a -pre.csv survey file")
    p_auto.add_argument("--out", required=True, help="output calibration model path (.npz)")
    p_auto.add_argument("--window-size", type=int, nargs="+", default=[3000, 1500, 750, 300],
                         help="candidate window size(s) in samples, tried widest-first (default "
                              "3000 1500 750 300, ~5min down to ~30s at 10Hz). How large a window "
                              "can be before real spatial field gradient contaminates it is a "
                              "property of the survey site -- if a wide one fails, progressively "
                              "shorter ones are tried automatically. Give one value to disable "
                              "fallback (see manual 8.7).")
    p_auto.add_argument("--step", type=int, default=None, help="stride between windows (default: window-size // 4)")
    p_auto.add_argument("--method", choices=["nearest", "spline"], default="nearest")
    p_auto.add_argument("--coverage-threshold-deg", type=float, default=10.0)
    p_auto.add_argument("--grid-resolution-deg", type=float, default=5.0)
    p_auto.add_argument("--max-variation", type=float, default=2.0,
                         help="nT peak-to-peak self-compensation pass/fail threshold (default 2.0)")
    p_auto.add_argument("--outlier-mad-threshold", type=float, default=3.0)
    p_auto.add_argument("--max-candidates", type=int, default=15,
                         help="max number of segments to accept and combine before giving up on "
                              "this window size")
    p_auto.add_argument("--candidate-pool-size", type=int, default=60,
                         help="how many scan-ranked windows to consider per window size (kept much "
                              "larger than --max-candidates: the top-ranked windows tend to cluster "
                              "as near-duplicate slices of whichever single region is cleanest, so a "
                              "shallow pool can get stuck unable to reach other parts of the survey, "
                              "see manual 8.7)")
    p_auto.add_argument("--max-overlap-fraction", type=float, default=0.3,
                         help="skip a candidate that overlaps an already-accepted segment by more "
                              "than this fraction (avoids near-duplicate slices of the same turn)")
    p_auto.add_argument("--coverage-target", type=float, default=0.1,
                         help="stop once the combined calibration's uncovered-heading fraction is at "
                              "or below this (default 0.1 = 10%%)")
    p_auto.add_argument("--coverage-sample-limit", type=int, default=20000,
                         help="estimate heading coverage from at most this many randomly chosen "
                              "survey samples while searching (0 = use every sample). The final "
                              "reported metrics and the pass/fail verdict are always recomputed "
                              "exactly on the full survey, so this only trades search speed")
    p_auto.add_argument("--export-vendor-cal", help="also write a vendor-format -cal.csv if it passes")
    p_auto.add_argument("--force-save", action="store_true",
                         help="save the best-effort model to --out even if it did not pass -- review "
                              "the printed quality report first")
    p_auto.set_defaults(func=_cmd_auto_calibrate)

    p_plot2d = sub.add_parser("plot2d", help="save a 2D spatial map (raw, or raw vs. compensated)")
    p_plot2d.add_argument("--pre", required=True, help="path to a -pre.csv or -comp.csv survey file")
    p_plot2d.add_argument("--out", required=True, help="output PNG path")
    p_plot2d.add_argument("--raw-col", default="Mag")
    p_plot2d.add_argument("--comp-col", default="MagComp",
                           help="compensated column to compare against; skipped if not present")
    p_plot2d.add_argument("--resolution-m", type=float, default=2.0)
    p_plot2d.set_defaults(func=_cmd_plot2d)

    p_plotheading = sub.add_parser("plot-heading", help="save a polar/azimuth heading time series plot")
    p_plotheading.add_argument("--pre", required=True, help="path to a -pre.csv survey file")
    p_plotheading.add_argument("--out", required=True, help="output PNG path")
    p_plotheading.set_defaults(func=_cmd_plot_heading)

    p_plotcal = sub.add_parser("plot-calibration", help="save a calibration self-compensation quality plot")
    p_plotcal.add_argument("--cal", required=True, help="path to a calibration model (.npz)")
    p_plotcal.add_argument("--out", required=True, help="output PNG path")
    p_plotcal.set_defaults(func=_cmd_plot_calibration)

    p_plotts = sub.add_parser(
        "plot-timeseries",
        help="save a whole-survey raw-vs-compensated time series plot (vendor Module 4 style)",
    )
    p_plotts.add_argument("--pre", required=True, help="path to a -pre-comp.csv (or -pre.csv) file")
    p_plotts.add_argument("--out", required=True, help="output PNG path")
    p_plotts.add_argument("--raw-col", default="Mag")
    p_plotts.add_argument("--comp-col", default="MagComp")
    p_plotts.set_defaults(func=_cmd_plot_timeseries)

    p_gui = sub.add_parser("gui", help="launch the desktop GUI (Tkinter)")
    p_gui.add_argument("--pre", default=None, help="optional -pre.csv to preload")
    p_gui.set_defaults(func=_cmd_gui)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
