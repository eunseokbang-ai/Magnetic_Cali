from .heading import compute_heading
from .io import comp_output_path, write_pre_csv


def apply_compensation(df_survey, model, mag_col="Mag", keep_raw=True):
    """Apply a CalibrationModel to a survey DataFrame.

    Adds a `MagComp` column with the compensated total field. When keep_raw is
    False, `mag_col` is overwritten with the compensated value instead (matching
    the vendor program's default behavior when "Keep Raw Data" is not checked).
    """
    theta, phi = compute_heading(
        df_survey["CompassX"].to_numpy(),
        df_survey["CompassY"].to_numpy(),
        df_survey["CompassZ"].to_numpy(),
    )
    delta_b = model.predict(theta, phi)
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
