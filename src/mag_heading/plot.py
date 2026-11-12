import matplotlib

matplotlib.use("Agg")  # headless-safe; Streamlit and file output both work off Figure objects

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

from .heading import compute_heading

EARTH_RADIUS_M = 6371000.0


def local_enu(lat, lon, lat0=None, lon0=None):
    """Equirectangular local East/North projection, in meters.

    Adequate for survey-sized areas (up to a few km); not for large-scale mapping.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    lat0 = float(np.nanmean(lat)) if lat0 is None else lat0
    lon0 = float(np.nanmean(lon)) if lon0 is None else lon0

    lat0_rad = np.radians(lat0)
    x = np.radians(lon - lon0) * EARTH_RADIUS_M * np.cos(lat0_rad)
    y = np.radians(lat - lat0) * EARTH_RADIUS_M
    return x, y


def grid_interpolate(x, y, values, resolution_m=2.0, method="cubic"):
    """Interpolate scattered (x, y, values) onto a regular grid.

    Falls back to 'linear' then 'nearest' to fill points 'cubic'/'linear' can't
    reach (outside the convex hull), matching how sparse survey-line data is
    usually handled -- cubic where well-supported, linear/nearest at the edges.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    values = np.asarray(values, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(values)
    x, y, values = x[valid], y[valid], values[valid]
    if len(x) < 3:
        raise ValueError("need at least 3 valid points to interpolate a 2D grid")

    grid_x = np.arange(x.min(), x.max() + resolution_m, resolution_m)
    grid_y = np.arange(y.min(), y.max() + resolution_m, resolution_m)
    gx, gy = np.meshgrid(grid_x, grid_y)

    points = np.column_stack([x, y])
    grid_z = griddata(points, values, (gx, gy), method=method)
    nan_mask = np.isnan(grid_z)
    if np.any(nan_mask):
        fallback = griddata(points, values, (gx, gy), method="nearest")
        grid_z[nan_mask] = fallback[nan_mask]

    return gx, gy, grid_z


def _mask_far_cells(gx, gy, x, y, max_distance_m):
    """NaN-out grid cells farther than max_distance_m from every sample, so the
    map doesn't show interpolation/extrapolation where nothing was flown."""
    from scipy.spatial import cKDTree

    valid = np.isfinite(x) & np.isfinite(y)
    tree = cKDTree(np.column_stack([x[valid], y[valid]]))
    dist, _ = tree.query(np.column_stack([gx.ravel(), gy.ravel()]))
    return (dist > max_distance_m).reshape(gx.shape)


def _robust_limits(*arrays, pct=2.0):
    vals = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays])
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(vals, [pct, 100.0 - pct])
    if hi <= lo:
        lo, hi = float(vals.min()), float(vals.max()) + 1e-9
    return float(lo), float(hi)


def plot_2d_comparison(df, raw_col="Mag", comp_col=None, resolution_m=2.0, title=None,
                       max_distance_m=20.0, clip_percentile=2.0, show_difference=True):
    """Raw vs. compensated 2D survey map, sharing a color scale (Zhang et al. Fig. 4/6 style).

    If comp_col is None or not present, only the raw panel is drawn. With a
    compensated column, a third panel shows raw - compensated (the heading
    correction that was removed) on a symmetric diverging scale -- stripes that
    follow flight direction there are the heading effect itself.

    The shared color scale is clipped to the clip_percentile..(100-clip_percentile)
    range of the gridded values: a handful of spikes (take-off/landing, a
    metal object) would otherwise stretch the scale until the survey itself is
    one flat color. Cells farther than max_distance_m from any sample are left
    blank (None disables masking).
    Returns the Figure; caller decides whether to save it or embed it.
    """
    x, y = local_enu(df["Latitude"], df["Longitude"])
    gx, gy, z_raw = grid_interpolate(x, y, df[raw_col], resolution_m=resolution_m)
    far = _mask_far_cells(gx, gy, x, y, max_distance_m) if max_distance_m else None
    if far is not None:
        z_raw[far] = np.nan

    has_comp = comp_col is not None and comp_col in df.columns
    if has_comp:
        _, _, z_comp = grid_interpolate(x, y, df[comp_col], resolution_m=resolution_m)
        if far is not None:
            z_comp[far] = np.nan
        vmin, vmax = _robust_limits(z_raw, z_comp, pct=clip_percentile)
        n_panels = 3 if show_difference else 2
        fig, axes = plt.subplots(1, n_panels, figsize=(4.6 * n_panels, 5), sharex=True, sharey=True)
        panels = [(axes[0], z_raw, "Raw"), (axes[1], z_comp, "Compensated")]
    else:
        vmin, vmax = _robust_limits(z_raw, pct=clip_percentile)
        fig, ax = plt.subplots(figsize=(6, 5))
        axes = [ax]
        panels = [(ax, z_raw, "Raw")]

    im = None
    for ax, z, label in panels:
        im = ax.pcolormesh(gx, gy, z, vmin=vmin, vmax=vmax, cmap="turbo", shading="auto")
        ax.set_title(label)
        ax.set_xlabel("East (m)")
        ax.set_aspect("equal")
    axes[0].set_ylabel("North (m)")
    fig.colorbar(im, ax=[p[0] for p in panels], label="nT", shrink=0.85)

    if has_comp and show_difference:
        z_diff = z_raw - z_comp
        lim = _robust_limits(np.abs(z_diff), pct=clip_percentile)[1] or 1.0
        ax = axes[2]
        im_d = ax.pcolormesh(gx, gy, z_diff, vmin=-lim, vmax=lim, cmap="RdBu_r", shading="auto")
        ax.set_title("Raw - Compensated (removed)")
        ax.set_xlabel("East (m)")
        ax.set_aspect("equal")
        fig.colorbar(im_d, ax=ax, label="nT", shrink=0.85)

    if title:
        fig.suptitle(title)
    return fig


def plot_heading_timeseries(df, title=None):
    """Polar/azimuth heading vs. sample index (Zhang et al. Fig. 3b style)."""
    theta, phi = compute_heading(df["CompassX"], df["CompassY"], df["CompassZ"])
    idx = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(idx, np.degrees(theta), label="polar", linewidth=0.8)
    ax.plot(idx, np.degrees(phi), label="azimuth", linewidth=0.8)
    ax.set_xlabel("sample index")
    ax.set_ylabel("heading (deg)")
    ax.legend()
    if title:
        ax.set_title(title)
    return fig


def plot_compensation_timeseries(df, raw_col="Mag", comp_col="MagComp", title=None):
    """Whole-survey raw vs. compensated field reading over time (vendor Module 4
    "MagArrow Data Plot" style: raw and compensated overlaid on one time axis).
    This is the direct way to check whether compensation actually reduced
    swing/motion noise across a full flight -- self-compensation plots only ever
    show the calibration segment itself, not the survey it gets applied to.
    """
    if "Time" in df.columns:
        try:
            t = pd.to_timedelta(df["Time"].astype(str)).dt.total_seconds()
            x = (t - t.iloc[0]).to_numpy()
            xlabel = "time (s)"
        except (ValueError, TypeError):
            x = np.arange(len(df))
            xlabel = "sample index"
    else:
        x = np.arange(len(df))
        xlabel = "sample index"

    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(x, df[raw_col].to_numpy(), label="raw", linewidth=0.6, color="tab:blue")
    if comp_col in df.columns:
        ax.plot(x, df[comp_col].to_numpy(), label="compensated", linewidth=0.8, color="tab:red")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mag (nT)")
    ax.legend()
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_calibration_quality(model, title=None):
    """Raw vs. self-compensated calibration data over sample index, with heading
    overlaid on a secondary axis (Zhang et al. Fig. 5a / vendor Module 3 "Calibration
    Data Quality Check" style). A flat red curve indicates the calibration segment's
    field variation is well explained by heading alone.
    """
    idx = np.arange(len(model.mag_cal))
    residual = model.mag_cal - model.baseline - model.predict(model.theta_cal, model.phi_cal)

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(idx, model.mag_cal - model.baseline, label="raw (offset)", linewidth=0.8, color="tab:blue")
    ax1.plot(idx, residual, label="self-compensated", linewidth=1.0, color="tab:red")
    ax1.set_xlabel("sample index")
    ax1.set_ylabel("field reading, offset (nT)")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(idx, np.degrees(model.theta_cal), label="polar", linewidth=0.6, color="tab:green", alpha=0.5)
    ax2.plot(idx, np.degrees(model.phi_cal), label="azimuth", linewidth=0.6, color="tab:orange", alpha=0.5)
    ax2.set_ylabel("heading (deg)")
    ax2.legend(loc="upper right")

    if title:
        ax1.set_title(title)
    fig.tight_layout()
    return fig


def plot_heading_coverage(model, survey_df=None, max_points=20000, title=None):
    """Where the survey's headings sit relative to what the calibration covers.

    Azimuth (x) vs. polar angle (y). Survey samples are grey when the calibration
    has data within its coverage threshold and red when it doesn't; the
    calibration's own samples are drawn on top in blue. A red cluster is a heading
    the survey flew a lot that the correction would have to extrapolate to -- the
    picture behind the report's survey_uncovered_fraction number.
    """
    from .heading import compute_heading

    fig, ax = plt.subplots(figsize=(10, 4))
    if survey_df is not None:
        theta_s, phi_s = compute_heading(
            survey_df["CompassX"].to_numpy(),
            survey_df["CompassY"].to_numpy(),
            survey_df["CompassZ"].to_numpy(),
        )
        if len(theta_s) > max_points:
            pick = np.random.default_rng(0).choice(len(theta_s), max_points, replace=False)
            theta_s, phi_s = theta_s[pick], phi_s[pick]
        uncovered = model.coverage_distance(theta_s, phi_s) > model._coverage_chord
        ax.scatter(np.degrees(phi_s[~uncovered]), np.degrees(theta_s[~uncovered]), s=2,
                   color="0.7", label=f"survey, covered ({np.mean(~uncovered):.0%})", rasterized=True)
        ax.scatter(np.degrees(phi_s[uncovered]), np.degrees(theta_s[uncovered]), s=3,
                   color="tab:red", label=f"survey, NOT covered ({np.mean(uncovered):.0%})",
                   rasterized=True)
    ax.scatter(np.degrees(model.phi_cal), np.degrees(model.theta_cal), s=4, color="tab:blue",
               label="calibration", rasterized=True)
    ax.set_xlim(-180, 180)
    ax.set_xlabel("azimuth (deg)")
    ax.set_ylabel("polar (deg)")
    ax.legend(loc="upper right", markerscale=3, fontsize=8)
    ax.set_title(title or f"heading coverage (threshold {model.coverage_threshold_deg:g} deg)")
    fig.tight_layout()
    return fig
