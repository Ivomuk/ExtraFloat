"""
check_loan_episode_dataset_integrity.py
===========================================
Deliverable 2 of the episode-grain rebuild of Analysis 3. Reads the output
of build_loan_episode_capacity_dataset.py and reports, before anything else
is attempted: snapshot staleness (as bands, not deleted -- the mart's real
refresh frequency is unknown, so this is descriptive, feeding the
sensitivity cuts used in Deliverables 3-5), missingness, duplicate-key
checks, outcome-eligibility/censoring breakdown, and the exposure-tier
distribution.

Restated independently (one-way scripts/ layering convention): the 7-tier
exposure set and the age-band edges are frozen here, matching the
conventions used throughout this workstream, not imported from any other
script.

Usage:
    python scripts\\check_loan_episode_dataset_integrity.py ^
        --episode-dataset loan_episode_capacity_dataset.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXPOSURE_TIERS_UGX = [50_000, 100_000, 250_000, 350_000, 500_000, 750_000, 1_000_000]

AGE_BAND_EDGES = [-0.5, 7.5, 30.5, 60.5, 90.5, np.inf]
AGE_BAND_LABELS = ["0-7", "8-30", "31-60", "61-90", ">90"]

REQUIRED_COLS = [
    "disbursement_fid", "agent_msisdn", "loan_date", "disbursement_amount_ugx",
    "fundamentals_age_days", "label_eligible_30d", "label_eligibility_reason_30d",
]


def age_band_distribution(df: pd.DataFrame) -> pd.DataFrame:
    valid = df["fundamentals_age_days"].notna()
    bands = pd.cut(df.loc[valid, "fundamentals_age_days"], bins=AGE_BAND_EDGES, labels=AGE_BAND_LABELS)
    counts = bands.value_counts().reindex(AGE_BAND_LABELS, fill_value=0)
    out = counts.rename("n_episodes").rename_axis("age_band").reset_index()
    out["age_band"] = out["age_band"].astype(str)
    out["pct_of_matched"] = round(out["n_episodes"] / valid.sum() * 100, 2) if valid.sum() else np.nan
    n_unmatched = int((~valid).sum())
    out_row = pd.DataFrame([{"age_band": "no_matched_snapshot", "n_episodes": n_unmatched,
                              "pct_of_matched": np.nan}])
    return pd.concat([out, out_row], ignore_index=True)


def missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    rows = []
    for c in df.columns:
        n_missing = int(df[c].isna().sum())
        rows.append({"column": c, "n_missing": n_missing, "pct_missing": round(n_missing / n * 100, 2) if n else np.nan})
    return pd.DataFrame(rows)


def duplicate_check(df: pd.DataFrame) -> dict:
    n_total = len(df)
    n_unique = df["disbursement_fid"].nunique()
    return {"n_rows": n_total, "n_unique_disbursement_fid": n_unique, "n_duplicate_rows": n_total - n_unique}


def eligibility_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for reason, cell in df.groupby("label_eligibility_reason_30d", dropna=False):
        rows.append({"label_eligibility_reason_30d": reason, "n_episodes": len(cell),
                     "pct": round(len(cell) / len(df) * 100, 2)})
    out = pd.DataFrame(rows).sort_values("n_episodes", ascending=False).reset_index(drop=True)
    n_eligible = int((df["label_eligible_30d"] == 1).sum())
    print(f"  Overall label_eligible_30d==1: {n_eligible:,} / {len(df):,} ({n_eligible / len(df) * 100:.1f}%)")
    return out


def exposure_tier_distribution(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    known = set(EXPOSURE_TIERS_UGX)
    for tier in EXPOSURE_TIERS_UGX:
        cell = df[df["disbursement_amount_ugx"] == tier]
        rows.append({"exposure_tier_ugx": tier, "n_episodes": len(cell)})
    unknown = df[~df["disbursement_amount_ugx"].isin(known)]
    rows.append({"exposure_tier_ugx": "other/unknown", "n_episodes": len(unknown)})
    if len(unknown):
        print(f"  NOTE: {len(unknown):,} episode(s) have a disbursement_amount_ugx outside the known "
              f"7-tier set -- values present: {sorted(unknown['disbursement_amount_ugx'].unique())[:20]}")
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode-dataset", default="loan_episode_capacity_dataset.csv")
    ap.add_argument("--out-prefix", default="loan_episode_integrity")
    args = ap.parse_args(argv)

    path = Path(args.episode_dataset)
    if not path.exists():
        sys.exit(f"ERROR: {path} not found -- run build_loan_episode_capacity_dataset.py first.")
    df = pd.read_csv(path, low_memory=False)
    missing_req = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_req:
        sys.exit(f"ERROR: {path} is missing required column(s): {missing_req}")

    print(f"Episode dataset: {len(df):,} row(s), {df['agent_msisdn'].nunique():,} unique agent(s).\n")

    print("=" * 100)
    print("Fundamentals snapshot age -- bands (NOT deleted here; descriptive only)")
    print("=" * 100)
    age_dist = age_band_distribution(df)
    print(age_dist.to_string(index=False))
    age_dist.to_csv(f"{args.out_prefix}_age_bands.csv", index=False)

    print("\n" + "=" * 100)
    print("Missingness per column")
    print("=" * 100)
    miss = missingness_report(df)
    print(miss.to_string(index=False))
    miss.to_csv(f"{args.out_prefix}_missingness.csv", index=False)

    print("\n" + "=" * 100)
    print("disbursement_fid uniqueness")
    print("=" * 100)
    dup = duplicate_check(df)
    print(dup)
    if dup["n_duplicate_rows"] > 0:
        print(f"  WARNING: {dup['n_duplicate_rows']} duplicate disbursement_fid row(s) found.")

    print("\n" + "=" * 100)
    print("label_eligible_30d / label_eligibility_reason_30d breakdown")
    print("=" * 100)
    elig = eligibility_breakdown(df)
    print(elig.to_string(index=False))
    elig.to_csv(f"{args.out_prefix}_eligibility_breakdown.csv", index=False)

    print("\n" + "=" * 100)
    print("disbursement_amount_ugx vs. the known 7-tier exposure set")
    print("=" * 100)
    tiers = exposure_tier_distribution(df)
    print(tiers.to_string(index=False))
    tiers.to_csv(f"{args.out_prefix}_exposure_tiers.csv", index=False)


if __name__ == "__main__":
    main()
