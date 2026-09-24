"""
Direct, literal check: which whitelist agents are absent from the final
scored output file (e.g. output/engine_test_output.csv)?

Complementary to check_whitelist_blacklist_eval.py's PART A, which checks
coverage against --borrower-file (an engine *input*) and explains *why* an
agent might be invisible to scoring. This script skips that explanation
and just answers the literal question: is this whitelist agent's msisdn
present in the actual final output at all? If this script finds MORE
missing agents than PART A's borrower_history-missing count, that means
something drops whitelist agents further down the pipeline (after
borrower_history.csv) that PART A wouldn't catch -- worth investigating,
not reconciling away.

Usage:
    python scripts\\check_whitelist_not_scored.py ^
        --whitelist-file data\\whitelist_aug_20260804.csv ^
        --output-file output\\engine_test_output.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--whitelist-file", required=True, metavar="PATH")
    p.add_argument("--output-file", required=True, metavar="PATH",
                   help="The scored output CSV (e.g. output/engine_test_output.csv).")
    p.add_argument("--out-missing", default=None, metavar="PATH",
                   help="Optional: write the missing whitelist agents to this CSV for inspection.")
    args = p.parse_args(argv)

    wl_path = Path(args.whitelist_file)
    out_path = Path(args.output_file)
    if not wl_path.exists():
        sys.exit(f"ERROR: file not found: {wl_path}")
    if not out_path.exists():
        sys.exit(f"ERROR: file not found: {out_path}")

    wl = pd.read_csv(wl_path)
    scored = pd.read_csv(out_path)

    if "agent_msisdn" not in wl.columns:
        sys.exit(f"ERROR: 'agent_msisdn' column not found in {wl_path} (columns: {list(wl.columns)})")
    if "msisdn" not in scored.columns:
        sys.exit(f"ERROR: 'msisdn' column not found in {out_path} (columns: {list(scored.columns)})")

    wl = wl.copy()
    wl["_key"] = wl["agent_msisdn"].astype(str).str.strip()
    scored_keys = set(scored["msisdn"].astype(str).str.strip())

    missing_mask = ~wl["_key"].isin(scored_keys)
    missing = wl.loc[missing_mask].drop(columns="_key")

    n_wl = len(wl)
    n_missing = len(missing)
    print(f"Loaded {n_wl:,} whitelist agents from {wl_path}")
    print(f"Loaded {len(scored):,} scored rows from {out_path}")
    print(f"\n{n_missing:,} of {n_wl:,} whitelist agents ({n_missing / n_wl:.1%}) "
          f"NOT present in the final scored output.")

    if n_missing > 0:
        sample_cols = [c for c in ("agent_msisdn", "agent_category", "reason") if c in missing.columns]
        print(f"\nSample (up to 10):\n{missing[sample_cols].head(10).to_string(index=False)}")

        if args.out_missing:
            out_missing_path = Path(args.out_missing)
            out_missing_path.parent.mkdir(parents=True, exist_ok=True)
            missing.to_csv(out_missing_path, index=False)
            print(f"\nWrote {n_missing:,} missing whitelist agents to {out_missing_path}")


if __name__ == "__main__":
    main()
