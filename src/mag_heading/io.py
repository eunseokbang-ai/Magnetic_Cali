import csv
import re

import pandas as pd

REQUIRED_PRE_COLUMNS = ["Mag", "CompassX", "CompassY", "CompassZ"]
REQUIRED_CAL_COLUMNS = ["Mag", "CompassX", "CompassY", "CompassZ"]


def _named_column_count(path):
    """Number of non-empty header fields in a -pre.csv (the rest are blank padding)."""
    with open(path, "r", encoding="utf-8", newline="") as f:
        header = f.readline()
    fields = header.rstrip("\r\n").split(",")
    count = len(fields)
    while count > 0 and fields[count - 1].strip() == "":
        count -= 1
    return count


def read_pre_csv(path):
    """Read a vendor `-pre.csv` survey file.

    GgaSentence/RmcSentence fields embed commas inside single-quote delimiters
    (LabVIEW CSV export convention), so the single quote is used as the CSV
    quote character rather than pandas' default double quote.
    """
    n_named = _named_column_count(path)
    # index_col=False is required, not cosmetic: when a data row has exactly one
    # more field than the header (seen in some raw Survey Manager exports, before
    # any padding/LPF columns get added downstream), pandas' default parser
    # silently treats the first column as an unnamed index and shifts every
    # named column one position to the right -- e.g. "Mag" quietly binds to the
    # MagValid flag's values instead. No error, no warning, just wrong numbers.
    df = pd.read_csv(path, quotechar="'", skipinitialspace=True, index_col=False)
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in REQUIRED_PRE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing required columns {missing}")
    df.attrs["n_named_columns"] = n_named
    df.attrs["n_original_columns"] = len(df.columns)
    df.attrs["source_path"] = str(path)
    return df


def _is_padding_column(name):
    """True for the vendor's blank header fields, which pandas auto-names."""
    return str(name).startswith("Unnamed:")


def write_pre_csv(df, path):
    """Write a DataFrame back out in the same layout `read_pre_csv` expects.

    Reproduces the vendor's blank trailing-column padding and single-quote
    quoting for fields containing commas, so the file can be reopened by the
    vendor program's later modules (noise reduction, combination).

    A column is emitted as a blank header field exactly when pandas auto-named
    it `Unnamed: N`, which is precisely the set of blank fields in the vendor
    header. Deriving it from the names rather than from a column-index range is
    what makes the round trip stable: a `-comp.csv` ends with a real `MagComp`
    column *after* the padding, so an index-based rule (everything past the
    named prefix is padding) either blanked out MagComp's name or -- reading the
    file back, where the trailing field is no longer blank -- leaked literal
    `Unnamed: 35,Unnamed: 36,...` text into the header on the next write.
    """
    columns = list(df.columns)
    header_fields = ["" if _is_padding_column(name) else str(name) for name in columns]
    header_line = ",".join(header_fields)

    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(header_line + "\n")
        df.to_csv(
            f,
            header=False,
            index=False,
            quotechar="'",
            quoting=csv.QUOTE_MINIMAL,
        )


def read_cal_csv(path):
    """Read a vendor-style `-cal.csv` (Mag, CompassX, CompassY, CompassZ only)."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in REQUIRED_CAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing required columns {missing}")
    return df


def write_cal_csv(df, path):
    df[REQUIRED_CAL_COLUMNS].to_csv(path, index=False)


def resolve_row_index(df, value):
    """Resolve a --start/--end CLI value to a row position.

    Accepts either an integer row index, or an "HH:MM:SS[.sss]" time string
    matched against the nearest row in the `Time` column.
    """
    try:
        return int(value)
    except ValueError:
        pass

    if "Time" not in df.columns:
        raise ValueError(f"cannot resolve '{value}': no Time column in data")
    if not re.match(r"^\d{1,2}:\d{2}:\d{2}(\.\d+)?$", str(value)):
        raise ValueError(f"'{value}' is neither an integer index nor an HH:MM:SS time")

    times = pd.to_timedelta(df["Time"].astype(str))
    target = pd.to_timedelta(str(value))
    return int((times - target).abs().idxmin())


def select_segment(df, start, end):
    start_idx = resolve_row_index(df, start)
    end_idx = resolve_row_index(df, end)
    if start_idx > end_idx:
        start_idx, end_idx = end_idx, start_idx
    segment = df.iloc[start_idx : end_idx + 1].reset_index(drop=True)
    segment.attrs.update(df.attrs)
    return segment


def select_segments(df, ranges, mag_col="Mag", center_segments=True):
    """Concatenate several (start, end) ranges from `df` into one calibration segment.

    A real vendor-built calibration file that's known to work well (Hongcheon
    project, verified by re-deriving it here) turned out to be assembled from
    many short, separately-picked, non-contiguous snippets rather than one long
    continuous window -- its row order jumps back and forth through the source
    file. A single window has to be simultaneously clean (low real-field
    gradient) *and* heading-diverse throughout its whole length, which is a hard
    combination to find in ordinary survey data; several short snippets only
    need to be individually clean, and their headings add up.

    Segments picked from different points in a long survey are usually also at
    different physical locations, so their real ambient field level differs --
    confirmed on the Hongcheon data, where four individually-excellent 150-sample
    segments (self p2p 0.46-0.58 nT each) had `Mag` medians spanning ~12 nT
    between segments. `CalibrationModel` fits one global baseline over the pooled
    samples, so that ~12 nT spatial spread gets misattributed to heading and
    wrecks the fit (this is exactly the vendor manual's "an offset is likely
    needed if you combine data points from two well separated areas" warning,
    just triggered within one file instead of across flights). `center_segments`
    removes each segment's own median before pooling so only within-segment,
    heading-driven variation survives; a single segment is unaffected (its own
    median offset is zero by construction), so this is safe to default on.
    """
    if not ranges:
        raise ValueError("ranges must contain at least one (start, end) pair")
    pieces = [select_segment(df, start, end) for start, end in ranges]

    offsets = [0.0] * len(pieces)
    if center_segments and len(pieces) > 1:
        medians = [float(p[mag_col].median()) for p in pieces]
        reference = sum(medians) / len(medians)
        offsets = [reference - m for m in medians]
        for piece, offset in zip(pieces, offsets):
            piece[mag_col] = piece[mag_col] + offset

    combined = pd.concat(pieces, ignore_index=True)
    combined.attrs.update(df.attrs)
    if center_segments and len(pieces) > 1:
        combined.attrs["segment_baseline_offsets_nT"] = offsets
    return combined


def comp_output_path(pre_path):
    pre_path = str(pre_path)
    if pre_path.lower().endswith(".csv"):
        return pre_path[: -len(".csv")] + "-comp.csv"
    return pre_path + "-comp.csv"
