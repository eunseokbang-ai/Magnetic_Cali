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


def batch_compensate(pre_paths, model, mag_col="Mag", keep_raw=True, out_dir=None):
    """Apply one calibration model to many -pre.csv files and report before/after
    noise reduction per file, so a calibration's real-world effect can be checked
    across an entire field survey before committing to it (rather than trusting a
    single candidate segment's self-compensation score alone).
    """
    comp_col = "MagComp" if keep_raw else mag_col
    results = []

    for path in pre_paths:
        df = read_pre_csv(path)
        compensated = apply_compensation(df, model, mag_col=mag_col, keep_raw=keep_raw)

        if out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, os.path.basename(comp_output_path(path)))
        else:
            out_path = None
        out_path = write_comp_csv(compensated, path, out_path=out_path)

        raw_metrics = compute_noise_metrics(df[mag_col].to_numpy())
        comp_metrics = compute_noise_metrics(compensated[comp_col].to_numpy())

        raw_noise = raw_metrics["motion_noise_std_nT"]
        comp_noise = comp_metrics["motion_noise_std_nT"]
        improvement_pct = 100.0 * (1.0 - comp_noise / raw_noise) if raw_noise > 0 else 0.0

        results.append(
            {
                "path": str(path),
                "out_path": str(out_path),
                "n_samples": int(len(df)),
                "raw_motion_noise_std_nT": raw_noise,
                "comp_motion_noise_std_nT": comp_noise,
                "motion_noise_improvement_pct": improvement_pct,
                "raw_p2p_nT": raw_metrics["p2p_nT"],
                "comp_p2p_nT": comp_metrics["p2p_nT"],
            }
        )

    return results
