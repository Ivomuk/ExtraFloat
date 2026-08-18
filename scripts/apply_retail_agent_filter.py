"""
Retail-agent filter -- structural, profile-based exclusion ONLY.

Design principle (established this session, do not violate): activity
(customers_served_3m or any other transaction-count signal) is NEVER used
as an exclusion criterion here. 35.5% of the core, unambiguously-retail
"Agent Silver Class" population is also zero-activity in the last 3
months (new/dormant agents, not non-retail ones) -- a pure activity
cutoff would wrongly strip out a third of the real retail book. This
filter only classifies by agent_profile (the account TYPE), which is a
structural fact about the account, not a behavioral one. Activity stays
available as a separate dormancy/thin-file signal downstream -- this
filter does not touch it.

Classification, built from this session's full-population evidence
(customers_served_3m by agent_profile, the commission-mismatch bucket
profiling, and direct business confirmation):

  RETAIL_PROFILES      -- "Agent {Bronze,Silver,Gold} Class" / "...Commission"
                          variants. ~95% of the file (Agent Silver Class
                          alone). Kept unconditionally, regardless of
                          activity level.
  NON_RETAIL_PROFILES  -- confirmed 100%-or-near-100%-zero-customers-served
                          niche/administrative profiles (NO END USER
                          TRANSACTIONS ALLOWED, Direct Sales, Merchant
                          Service Centers, Territory Master variants,
                          Referral/Retainer/CDP/DSD/AH niche profiles),
                          plus two aggregator/API-scale profiles (Service
                          Provider Agency Profile, Open API Agent CashIn
                          CashOut profile) identified by implausibly LARGE
                          customer counts rather than zero. Always excluded.
  REBALANCER_PROFILES  -- Super Agent / Master Agent family. Business
                          confirmed these primarily rebalance float between
                          agents, "sometimes do cash-in/cash-out here and
                          there" -- matches the customers_served_3m evidence
                          (low but not always zero). Excluded by default
                          per the "only retail agents" scope; set
                          --include-rebalancers to keep them if that
                          decision changes.
  DERISK_PROFILES      -- "Agent Derisk Class" specifically. NOT yet
                          confirmed by business whether this means
                          risk-restricted-but-still-retail (keep) or a
                          fundamentally different handling track (exclude).
                          Defaults to excluded (conservative, matches the
                          "only retail agents" directive) -- set
                          --include-derisk to keep it once confirmed.

Anything not matching any list above is UNCLASSIFIED, not silently kept:
excluded by default and printed loudly, since this session's profile
enumeration (~36 distinct values seen across different sub-populations)
is not guaranteed exhaustive against the full file. Extend the lists
above once a new unclassified value is confirmed one way or the other.

Usage:
    python apply_retail_agent_filter.py ^
        --transaction-file data\\mfs_daily_agent_mart_20260731.csv ^
        --profile-col agent_profile

Add --include-rebalancers and/or --include-derisk to keep those groups
once business confirms they should count as retail.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

RETAIL_PROFILES = {
    "agent bronze class",
    "agent silver class",
    "agent gold class",
    "agent bronze commission",
    "agent silver commission",
    "agent gold commission",
}

NON_RETAIL_PROFILES = {
    "no end user transactions allowed",
    "direct sales",
    "merchant service centers",
    "merchant master agent hierarchy account",
    "super agent escrow master account",
    "super agent escrow branch account",
    "super agent branch handler",
    "ah super agent branch handler",
    "ah agent bronze class",
    "ah service centers",
    "agent aggregated territory master",
    "agent independent territory master",
    "referralcampaignbonusprofile",
    "agent retainer",
    "cdp bronze agent profile",
    "dealer branch dsd agent profile",
    "dsd bronze agent profile",
    "dsd silver agent profile",
    "mobile money dsd agent profile",
    "service provider agency profile",
    "open api agent cashin cashout profile",
}

REBALANCER_PROFILES = {
    "super agent master account",
    "super agent branch account",
    "super agent commission",
    "ah super agent master account",
    "master agent silver class",
    "master agent bronze class",
    "master agent silver commission",
    "master agent commission",
}

DERISK_PROFILES = {
    "agent derisk class",
}


def classify_agent_profile(profile: str, include_rebalancers: bool, include_derisk: bool) -> str:
    """Returns one of: retail, non_retail, rebalancer, derisk, unclassified."""
    key = str(profile).strip().lower()
    if key in RETAIL_PROFILES:
        return "retail"
    if key in NON_RETAIL_PROFILES:
        return "non_retail"
    if key in REBALANCER_PROFILES:
        return "rebalancer"
    if key in DERISK_PROFILES:
        return "derisk"
    return "unclassified"


def is_retail(profile: str, include_rebalancers: bool = False, include_derisk: bool = False) -> bool:
    """The actual filter predicate -- True iff this agent_profile should be
    treated as retail for scoring purposes, given the current business
    decisions on the two conditional groups."""
    cat = classify_agent_profile(profile, include_rebalancers, include_derisk)
    if cat == "retail":
        return True
    if cat == "rebalancer" and include_rebalancers:
        return True
    if cat == "derisk" and include_derisk:
        return True
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--transaction-file", default="data/mfs_daily_agent_mart_20260731.csv")
    ap.add_argument("--profile-col", default="agent_profile")
    ap.add_argument("--include-rebalancers", action="store_true",
                     help="Keep Super Agent / Master Agent profiles as retail (default: excluded)")
    ap.add_argument("--include-derisk", action="store_true",
                     help="Keep Agent Derisk Class as retail (default: excluded, pending business confirmation)")
    ap.add_argument("--out-retail", default="retail_agents_filtered.csv")
    ap.add_argument("--out-excluded", default="retail_agents_excluded.csv")
    args = ap.parse_args()

    txn_path = Path(args.transaction_file)
    if not txn_path.exists():
        sys.exit(f"ERROR: transaction file not found: {txn_path}")

    df = pd.read_csv(txn_path, sep=",")
    if args.profile_col not in df.columns:
        sys.exit(f"ERROR: '{args.profile_col}' column not found. Found: {list(df.columns)[:30]}")

    df["_retail_category"] = df[args.profile_col].apply(
        lambda p: classify_agent_profile(p, args.include_rebalancers, args.include_derisk)
    )
    df["_is_retail"] = df[args.profile_col].apply(
        lambda p: is_retail(p, args.include_rebalancers, args.include_derisk)
    )

    n_total = len(df)
    print(f"Total agents: {n_total:,}\n")
    print("=== Classification breakdown ===")
    cat_counts = df["_retail_category"].value_counts()
    for cat, n in cat_counts.items():
        kept = "KEPT" if (
            cat == "retail"
            or (cat == "rebalancer" and args.include_rebalancers)
            or (cat == "derisk" and args.include_derisk)
        ) else "excluded"
        print(f"  {cat:15s} {n:>9,}  ({n/n_total:.1%})  -- {kept}")

    unclassified = df[df["_retail_category"] == "unclassified"]
    if len(unclassified) > 0:
        n_unclassified_profiles = unclassified[args.profile_col].nunique()
        print(
            f"\n*** {len(unclassified):,} agents ({len(unclassified)/n_total:.1%}) have an "
            f"agent_profile not in any known list -- excluded by default, not silently kept. ***"
        )
        print(f"Distinct unclassified profile values ({n_unclassified_profiles}):")
        print(unclassified[args.profile_col].value_counts().to_string())
        print("Review these and add them to RETAIL_PROFILES / NON_RETAIL_PROFILES / "
              "REBALANCER_PROFILES / DERISK_PROFILES in this script as appropriate.")

    retail_df = df[df["_is_retail"]].drop(columns=["_retail_category", "_is_retail"])
    excluded_df = df[~df["_is_retail"]]

    print(f"\nRetail population (kept): {len(retail_df):,} ({len(retail_df)/n_total:.1%})")
    print(f"Excluded population: {len(excluded_df):,} ({len(excluded_df)/n_total:.1%})")

    retail_df.to_csv(args.out_retail, index=False)
    excluded_df.to_csv(args.out_excluded, index=False)
    print(f"\nRetail-only extract written to: {args.out_retail}")
    print(f"Excluded agents (with _retail_category reason) written to: {args.out_excluded}")


if __name__ == "__main__":
    main()
