"""
Summarizes the diag_* research columns produced by a segmentation run with
clustering.enable_diagnostics=True (e.g. via run_segmentation_diagnostics.bat)
-- these never feed capacity_tier, so this is purely for a human analyst to
see what the GMM/HDBSCAN research layer actually found.

Reports:
  - diag_hdb_tier distribution (HDBSCAN-derived activity tiers, descriptive
    labels only -- see _HDB_TIER_NAMES in extrafloat_segmentation_pipeline.py)
  - diag_ensemble_cluster distribution
  - diag_is_anomaly count/rate, and how anomaly rate varies by capacity_tier
    (an agent can be e.g. Gold + anomalous at the same time -- this never
    changes their tier, but is worth a manual look)
  - cross-tab: capacity_tier (the real, deterministic business tier) vs.
    diag_hdb_tier (the research-only clustering tier) -- do they broadly
    agree, or does the research layer surface a meaningfully different
    grouping?

Usage:
    python scripts\\summarize_diagnostic_clustering.py --agents segmentation_outputs_diagnostics\\agent_segments.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agents", required=True, metavar="PATH")
    args = p.parse_args(argv)

    path = Path(args.agents)
    if not path.exists():
        sys.exit(f"ERROR: file not found: {path}")

    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} agent rows from {path}\n")

    diag_cols = [c for c in df.columns if c.startswith("diag_")]
    if not diag_cols:
        sys.exit(
            "ERROR: no diag_* columns found in this file -- was it produced with "
            "clustering.enable_diagnostics=True and --keep-intermediate?"
        )
    print(f"diag_* columns present: {diag_cols}\n")

    if "diag_hdb_tier" in df.columns:
        print("=== diag_hdb_tier distribution ===")
        print(df["diag_hdb_tier"].value_counts(dropna=False).to_string())
        print()

    if "diag_ensemble_cluster" in df.columns:
        print("=== diag_ensemble_cluster distribution ===")
        print(df["diag_ensemble_cluster"].value_counts(dropna=False).to_string())
        print()

    if "diag_is_anomaly" in df.columns:
        n_anomaly = int(df["diag_is_anomaly"].fillna(False).astype(bool).sum())
        print(f"=== diag_is_anomaly: {n_anomaly:,} of {len(df):,} agents ({n_anomaly / len(df):.1%}) ===\n")
        if "capacity_tier" in df.columns:
            print("Anomaly rate by capacity_tier (does anomaly flagging skew toward any tier?):")
            rate_by_tier = (
                df.assign(_anom=df["diag_is_anomaly"].fillna(False).astype(bool))
                .groupby("capacity_tier")["_anom"]
                .agg(["sum", "count"])
            )
            rate_by_tier["rate"] = rate_by_tier["sum"] / rate_by_tier["count"]
            print(rate_by_tier.sort_values("rate", ascending=False).to_string())
            print()

    if "capacity_tier" in df.columns and "diag_hdb_tier" in df.columns:
        print("=== capacity_tier (real, deterministic) vs. diag_hdb_tier (research-only) ===")
        print(pd.crosstab(df["capacity_tier"], df["diag_hdb_tier"], dropna=False).to_string())
        print()


if __name__ == "__main__":
    main()
