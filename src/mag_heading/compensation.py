import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from .heading import compute_heading
from .io import comp_output_path, write_pre_csv

DEFAULT_SMOOTH_ORDER = 5


def infer_sample_rate_hz(df):
    """Sample rate from the `Time` column, or None if it can't be determined."""
    if "Time" not in df.columns or len(df) < 2:
        return None
    try:
        t = pd.to_timedelta(df["Time"].astype(str)).dt.total_seconds().to_numpy()
    except (ValueError, TypeError):
        return None
    dt = np.median(np.diff(t))
    if not np.isfinite(dt) or dt <= 0:
        return None
    return float(1.0 / dt)


def smooth_delta_b(delta_b, smooth_hz, sample_rate_hz, order=DEFAULT_SMOOTH_ORDER):
    """Low-pass the heading correction before it is subtracted.

    The correction is computed per sample from the compass, so it inherits the
    compass's full-bandwidth noise, while the heading effect it is meant to
    remove varies only as fast as the sensor's attitude does -- well under 1 Hz.
    Subtracting the unsmoothed correction therefore injects noise: measured on
    the 2026-09 M350 survey flights, delta_b's own sample-to-sample std (0.265
    nT on ACQU51) matched the Mag channel's (0.263 nT), and compensation made
    the data *worse* by up to 27%. Low-passing delta_b at 1 Hz turned every one
    of those files positive (ACQU51 -27.4% -> +6.1%). This is the practical form
    of the compass-noise limit Zhang et al. (2022) state as
    "compass noise density < N * B / C" for a heading-effect slope C.

    Cutoff guidance: 1 Hz worked best across the 2026-09 set; 0.2 Hz (the
    vendor's own Mag filter default) over-smooths and throws the correction away.
    """
    nyquist = sample_rate_hz / 2.0
    if not 0 < smooth_hz < nyquist:
        raise ValueError(
            f"smooth_hz ({smooth_hz}) must be between 0 and the Nyquist frequency "
            f"({nyquist}) for a {sample_rate_hz} Hz sample rate"
        )
    b, a = butter(order, smooth_hz / nyquist, btype="low")
    return filtfilt(b, a, delta_b)


def apply_compensation(
    df_survey, model, mag_col="Mag", keep_raw=True, smooth_hz=None, sample_rate_hz=None
):
    """Apply a CalibrationModel to a survey DataFrame.

    Adds a `MagComp` column with the compensated total field. When keep_raw is
    False, `mag_col` is overwritten with the compensated value instead (matching
    the vendor program's default behavior when "Keep Raw Data" is not checked).

    `smooth_hz` low-passes the correction before subtracting it, which is what
    keeps compass noise from being injected into the compensated field -- see
    `smooth_delta_b` for the measurements behind this. It is off by default so
    that previously validated calibrations keep reproducing exactly; 1.0 (Hz) is
    the recommended starting value for 10 Hz survey data. `sample_rate_hz` is
    inferred from the `Time` column when not given.
    """
    theta, phi = compute_heading(
        df_survey["CompassX"].to_numpy(),
        df_survey["CompassY"].to_numpy(),
        df_survey["CompassZ"].to_numpy(),
    )
    delta_b = model.predict(theta, phi)

    if smooth_hz is not None:
        rate = sample_rate_hz or infer_sample_rate_hz(df_survey)
        if rate is None:
            raise ValueError(
                "smooth_hz needs a sample rate, but it could not be inferred from a "
                "'Time' column -- pass sample_rate_hz explicitly"
            )
        delta_b = smooth_delta_b(delta_b, smooth_hz, rate)

    mag_comp = df_survey[mag_col].to_numpy() - delta_b

    out = df_survey.copy()
    out.attrs.update(df_survey.attrs)
    if keep_raw:
        out["MagComp"] = mag_comp
    else:
        out[mag_col] = mag_comp
    return out


def write_comp_csv(df, original_path, out_path=None):
    out_path = out_path or comp_output_path(original_path)
    write_pre_csv(df, out_path)
    return out_path
