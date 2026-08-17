"""
Measure how widespread masked/truncated msisdn values are across the files
used this session -- e.g. "2567" (country code + leading digit, with every
subsequent digit stripped), which is NOT a unique agent identifier and would
silently corrupt any join or drop_duplicates() keyed on msisdn.

For each file given, reports:
  - the length distribution of the normalized msisdn column (a real Ugandan
    MSISDN should be a consistent, long digit count; masked values will show
    up as a short-length outlier cluster)
  - the most frequent normalized msisdn values and their row counts (a
    legitimate per-agent identifier should appear at most a small number of
    times per file; a value appearing hundreds or thousands of times is
    almost certainly a shared placeholder, not a real distinct agent)
  - how many rows fall below a configurable "looks masked" length threshold

Usage:
    python check_masked_msisdn_scope.py ^
        --transaction-file data\\mfs_daily_agent_mart_20260731.csv --transaction-col agent_msisdn ^
        --borrower-file data\\borrower_history.csv --borrower-col msisdn ^
        --output-file output\\engine_test_output.csv --output-col msisdn ^
        --fin-log-file data\\fin_log_cfw_202607211227.csv --fin-log-col instruct_to_fro_msisdn

All four file args are optional -- pass only the ones you want checked.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def _norm(s: pd.Series) -> pd.Series:
    out = s.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    return out.mask(out.str.lower().isin({"", "nan", "none", "<na>"}))


def analyze_file(label: str, path: str, col: str, short_len_threshold: int, top_n: int):
    p = Path(path)
    if not p.exists():
        print(f"[{label}] file not found: {p} -- skipping.")
        return

    df = pd.read_csv(p, sep=",")
    if col not in df.columns:
        print(f"[{label}] column '{col}' not found. Columns present: {list(df.columns)[:20]}")
        return

    vals = _norm(df[col])
    n_total = len(vals)
    n_null = int(vals.isna().sum())
    vals = vals.dropna()

    print(f"\n{'=' * 70}\n[{label}] column='{col}'  file={p}")
    print(f"Rows: {n_total:,}  (null/unparseable: {n_null:,})")

    lengths = vals.str.len()
    print("\nLength distribution of normalized values:")
    print(lengths.value_counts().sort_index().to_string())

    short = vals[lengths <= short_len_threshold]
    print(
        f"\nRows with length <= {short_len_threshold} (looks masked/truncated): "
        f"{len(short):,} ({len(short)/max(1,len(vals)):.2%})"
    )

    print(f"\nTop {top_n} most frequent normalized values (a real per-agent id "
          f"should appear only a handful of times at most):")
    top = vals.value_counts().head(top_n)
    print(top.to_string())

    # Flag: does the single most-frequent value look suspiciously dominant?
    if len(top) > 0:
        top_val, top_count = top.index[0], int(top.iloc[0])
        if top_count > 5:
            print(
                f"\n*** '{top_val}' appears {top_count:,} times -- almost certainly a shared "
                f"placeholder / masked value, not {top_count:,} instances of one real agent. ***"
            )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--transaction-file", default=None)
    ap.add_argument("--transaction-col", default="agent_msisdn")
    ap.add_argument("--borrower-file", default=None)
    ap.add_argument("--borrower-col", default="msisdn")
    ap.add_argument("--output-file", default=None)
    ap.add_argument("--output-col", default="msisdn")
    ap.add_argument("--fin-log-file", default=None)
    ap.add_argument("--fin-log-col", default="instruct_to_fro_msisdn")
    ap.add_argument("--short-len-threshold", type=int, default=6, help="Length at/below which a value is flagged as looking masked")
    ap.add_argument("--top-n", type=int, default=15)
    args = ap.parse_args()

    checks = [
        ("transaction", args.transaction_file, args.transaction_col),
        ("borrower", args.borrower_file, args.borrower_col),
        ("output", args.output_file, args.output_col),
        ("fin-log", args.fin_log_file, args.fin_log_col),
    ]
    ran_any = False
    for label, path, col in checks:
        if path:
            ran_any = True
            analyze_file(label, path, col, args.short_len_threshold, args.top_n)

    if not ran_any:
        sys.exit("ERROR: no file arguments given -- pass at least one of --transaction-file / --borrower-file / --output-file / --fin-log-file")


if __name__ == "__main__":
    main()
