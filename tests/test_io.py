import numpy as np
import pandas as pd
import pytest

from mag_heading.io import read_pre_csv, select_segment, select_segments


def test_read_pre_csv_handles_header_data_field_count_mismatch(tmp_path):
    """Regression test for a real bug found in raw Survey Manager exports: a data
    row with exactly one more field than the header makes pandas' default parser
    treat the first column as an implicit index, silently shifting every named
    column one position to the right (Mag ends up holding MagValid's values, etc).
    No exception is raised, so this must be caught by an explicit value check,
    not just "did it parse".
    """
    path = tmp_path / "mismatched-20Hz.csv"
    header = "Counter,Date,Time,Mag,MagValid,CompassX,CompassY,CompassZ\n"
    # exactly one extra trailing field vs the header (matches the real-world file)
    row = "1,2026-08-05,00:00:00.000,48734.41545,1,9730,33949,44645,\n"
    path.write_text(header + row, encoding="utf-8")

    df = read_pre_csv(path)

    assert df["Mag"].iloc[0] == 48734.41545
    assert df["CompassX"].iloc[0] == 9730
    assert df["CompassY"].iloc[0] == 33949
    assert df["CompassZ"].iloc[0] == 44645


def test_read_pre_csv_matched_field_counts_still_works(tmp_path):
    path = tmp_path / "matched-pre.csv"
    header = "Counter,Mag,CompassX,CompassY,CompassZ\n"
    row = "1,49809.86475,16738,-10602,54478\n"
    path.write_text(header + row, encoding="utf-8")

    df = read_pre_csv(path)

    assert df["Mag"].iloc[0] == 49809.86475
    assert df["CompassX"].iloc[0] == 16738


def test_write_pre_csv_round_trips_a_comp_file_header(tmp_path):
    # A -comp.csv ends with a real MagComp column *after* the vendor's blank
    # padding. Deriving the blank fields from a column-index range meant the
    # re-read file (whose last field is no longer blank) wrote literal
    # "Unnamed: N" text back into the header, breaking the vendor layout.
    from mag_heading.io import write_pre_csv

    path = tmp_path / "padded-pre.csv"
    header = "Counter,Mag,CompassX,CompassY,CompassZ,,,\n"
    path.write_text(header + "1,49809.9,16738,-10602,54478,,,\n", encoding="utf-8")

    df = read_pre_csv(path)
    df["MagComp"] = df["Mag"] - 1.0

    first = tmp_path / "out1.csv"
    write_pre_csv(df, first)
    second = tmp_path / "out2.csv"
    write_pre_csv(read_pre_csv(first), second)

    h1 = first.read_text(encoding="utf-8").splitlines()[0].split(",")
    h2 = second.read_text(encoding="utf-8").splitlines()[0].split(",")

    assert h1 == h2, "re-writing a -comp.csv must reproduce the same header"
    assert not any(f.startswith("Unnamed") for f in h2)
    assert h2[-1] == "MagComp"
    assert h2[5] == "" and h2[6] == "" and h2[7] == ""


def test_write_pre_csv_preserves_vendor_blank_padding(tmp_path):
    from mag_heading.io import write_pre_csv

    path = tmp_path / "vendor-pre.csv"
    path.write_text(
        "Counter,Mag,CompassX,CompassY,CompassZ,,,\n1,49809.9,16738,-10602,54478,,,\n",
        encoding="utf-8",
    )
    out = tmp_path / "rewritten.csv"
    write_pre_csv(read_pre_csv(path), out)

    assert out.read_text(encoding="utf-8").splitlines()[0] == "Counter,Mag,CompassX,CompassY,CompassZ,,,"


def _make_indexed_df(n=1000):
    return pd.DataFrame({
        "Mag": np.arange(n, dtype=float),
        "CompassX": np.arange(n, dtype=float),
        "CompassY": np.arange(n, dtype=float),
        "CompassZ": np.arange(n, dtype=float),
    })


def test_select_segments_concatenates_multiple_ranges_in_order():
    df = _make_indexed_df()
    combined = select_segments(df, [(100, 104), (500, 502), (10, 11)], center_segments=False)

    assert list(combined["Mag"]) == [100, 101, 102, 103, 104, 500, 501, 502, 10, 11]
    assert list(combined["CompassX"]) == [100, 101, 102, 103, 104, 500, 501, 502, 10, 11]


def test_select_segments_centers_each_segment_to_a_common_baseline():
    """Segments from different points in a survey usually sit at different real
    field levels; centering must remove that per-segment offset from Mag only
    (not the compass columns, which describe heading and must stay untouched)
    before pooling, so a downstream single global median-baseline is meaningful.
    """
    df = _make_indexed_df()
    combined = select_segments(df, [(100, 104), (500, 502)], center_segments=True)

    # each segment's own median should land on the same reference level
    first_segment_median = combined["Mag"].iloc[0:5].median()
    second_segment_median = combined["Mag"].iloc[5:8].median()
    assert first_segment_median == pytest.approx(second_segment_median)

    # heading-defining columns must be untouched by centering
    assert list(combined["CompassX"]) == [100, 101, 102, 103, 104, 500, 501, 502]

    assert "segment_baseline_offsets_nT" in combined.attrs
    assert len(combined.attrs["segment_baseline_offsets_nT"]) == 2


def test_select_segments_single_range_unaffected_by_centering():
    df = _make_indexed_df()
    combined = select_segments(df, [(100, 104)], center_segments=True)
    assert list(combined["Mag"]) == [100, 101, 102, 103, 104]


def test_select_segments_single_range_matches_select_segment():
    df = _make_indexed_df()
    single = select_segment(df, 50, 60)
    via_multi = select_segments(df, [(50, 60)])

    pd.testing.assert_frame_equal(
        single.reset_index(drop=True), via_multi.reset_index(drop=True)
    )


def test_select_segments_rejects_empty_ranges():
    df = _make_indexed_df()
    with pytest.raises(ValueError):
        select_segments(df, [])


def test_select_segments_preserves_attrs():
    df = _make_indexed_df()
    df.attrs["n_named_columns"] = 4
    df.attrs["source_path"] = "somefile.csv"
    combined = select_segments(df, [(0, 5), (20, 25)])
    assert combined.attrs["n_named_columns"] == 4
    assert combined.attrs["source_path"] == "somefile.csv"
