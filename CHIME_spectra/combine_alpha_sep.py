#!/usr/bin/env python3
"""
Cross-match baseband and CHIME/CAT1 spectral-index tables on frb_name,
then attach the sep_deg column from the crossmatch catalog.

Output columns:
    frb_name, dm,
    freq_top_base, freq_low_base, alpha_base, alpha_base_err,
    peak_freq_gauss_base, peak_freq_gauss_err_base,
    freq_top_cat1, freq_low_cat1, alpha_cat1, alpha_cat1_err,
    peak_freq_cat1, peak_freq_gauss_cat1, peak_freq_gauss_err_cat1,
    sep_deg

(column order follows BASEBAND_SRC_COLS / CAT1_SRC_COLS below --
add/reorder entries there to change what gets pulled in)
"""

import pandas as pd

# ----------------------------------------------------------------------
# File paths
# ----------------------------------------------------------------------
BASEBAND_CSV = "/home/thsu/FRB_population/CHIME_spectra/baseband_spectra/baseband_spectral_index_summary.csv"
CAT1_CSV = "/home/thsu/FRB_population/CHIME_spectra/chimecat1_spectra/chimecat1_alpha.csv"
CROSSMATCH_CSV = "/home/thsu/FRB_population/cross_matched_with_separation.csv"
OUTPUT_CSV = "/home/thsu/FRB_population/CHIME_spectra/combined_alpha_sep.csv"

# ----------------------------------------------------------------------
# Column-name mapping
# ----------------------------------------------------------------------
# Edit the lists below to match the ACTUAL column names in your two
# source CSVs. The script will print each file's columns at the start
# of the run so you can check/adjust these before the merge happens
# (it will stop with a clear error if a mapped name isn't found,
# rather than silently producing wrong data).
#
# To add more columns later: just add another entry to BASEBAND_SRC_COLS
# or CAT1_SRC_COLS. Every column gets suffixed automatically (_base or
# _cat1) unless it's listed in NO_SUFFIX_COLS (e.g. dm, which only
# exists in one file so doesn't need disambiguating).

FRB_NAME_COL = "frb_name"  # name of the FRB-name column in baseband/cat1 files (edit if different, e.g. "name")
CROSSMATCH_NAME_COL = "TNS_name"  # name of the FRB-name column in CROSSMATCH_CSV (it differs from the others)

# source column names *as they currently appear* in the baseband csv
BASEBAND_SRC_COLS = [
    "dm",
    "freq_top",
    "freq_low",
    "alpha",
    "alpha_err",
    "peak_freq_gauss",
    "peak_freq_gauss_err",
]

# source column names *as they currently appear* in the cat1 csv
CAT1_SRC_COLS = [
    "freq_top",
    "freq_low",
    "alpha",
    "alpha_err",
    "peak_freq",
    "peak_freq_gauss",
    "peak_freq_gauss_err",
]

# columns that should NOT get a _base/_cat1 suffix in the output
NO_SUFFIX_COLS = {"dm"}

BASEBAND_SUFFIX = "_base"
CAT1_SUFFIX = "_cat1"

SEP_COL = "sep_deg"  # column name in CROSSMATCH_CSV holding the separation


def suffixed(col, suffix):
    """Return the output column name for a source column."""
    return col if col in NO_SUFFIX_COLS else f"{col}{suffix}"


# built from the lists above -- {source_name: output_name}
BASEBAND_COLS = {c: suffixed(c, BASEBAND_SUFFIX) for c in BASEBAND_SRC_COLS}
CAT1_COLS = {c: suffixed(c, CAT1_SUFFIX) for c in CAT1_SRC_COLS}


def load_and_check(path, required_cols, label):
    df = pd.read_csv(path)
    print(f"[{label}] loaded {len(df)} rows from {path}")
    print(f"[{label}] columns: {list(df.columns)}")
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"[{label}] missing expected column(s) {missing}. "
            f"Available columns are: {list(df.columns)}. "
            f"Update the *_SRC / FRB_NAME_COL mapping at the top of the script."
        )
    return df


def main():
    # ---- load ----
    baseband_required = [FRB_NAME_COL] + list(BASEBAND_COLS.keys())
    cat1_required = [FRB_NAME_COL] + list(CAT1_COLS.keys())
    cross_required = [CROSSMATCH_NAME_COL, SEP_COL]

    base_df = load_and_check(BASEBAND_CSV, baseband_required, "baseband")
    cat1_df = load_and_check(CAT1_CSV, cat1_required, "cat1")
    cross_df = load_and_check(CROSSMATCH_CSV, cross_required, "crossmatch")
    cross_df = cross_df.rename(columns={CROSSMATCH_NAME_COL: FRB_NAME_COL})

    # ---- normalise / rename source columns to target names ----
    base_df = base_df.rename(columns=BASEBAND_COLS)[
        [FRB_NAME_COL] + list(BASEBAND_COLS.values())
    ]

    cat1_df = cat1_df.rename(columns=CAT1_COLS)[
        [FRB_NAME_COL] + list(CAT1_COLS.values())
    ]

    # dedupe crossmatch on frb_name, keep just what we need
    cross_df = cross_df[[FRB_NAME_COL, SEP_COL]].drop_duplicates(subset=FRB_NAME_COL)

    # ---- merge baseband + cat1 on frb_name ----
    # inner join: FRBs missing from either table are dropped (as requested).
    combined = pd.merge(base_df, cat1_df, on=FRB_NAME_COL, how="inner")
    print(f"[merge] {len(combined)} FRBs present in both baseband & cat1")

    # ---- attach sep_deg ----
    # inner join: FRBs with no crossmatch/sep_deg entry are also dropped.
    combined = pd.merge(combined, cross_df, on=FRB_NAME_COL, how="inner")
    print(f"[merge] {len(combined)} FRBs remain after also requiring a sep_deg match")

    # ---- reorder & save ----
    final_cols = (
        [FRB_NAME_COL]
        + list(BASEBAND_COLS.values())
        + list(CAT1_COLS.values())
        + [SEP_COL]
    )
    combined = combined[final_cols]
    combined.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(combined)} rows -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()