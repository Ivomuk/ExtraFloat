"""
compare_options.py
==================
Compare two float limit assignment architectures side by side.

Option A (Current):
    commission → agent_category → hard ceiling
    segmentation → capacity_tier is informational only

Option B (Bounded behavioural adjustment):
    commission → agent_category → base ceiling  (eligibility gate — cannot be bypassed)
    segmentation → capacity_tier adjusts ceiling ±1 tier from commission base
    anomaly flag → caps at one tier below agent_category ceiling, floor at New Bronze

Key design constraints in Option B:
  1. Commission is the eligibility gate. Below Threshold agents (commission < 50,000)
     receive 0 regardless of capacity_tier — behaviour cannot grant eligibility.
  2. capacity_tier can move the ceiling at most ±1 tier from the commission base.
     A Bronze agent with Diamond behaviour reaches Silver (not Diamond).
  3. Anomaly penalty floors at New Bronze (50,000) for commissioned agents.
     Unusual behaviour reduces limits but does not eliminate eligibility.

Usage:
    python compare_options.py
    python compare_options.py --input output/engine_test_output.csv --output output/options_comparison.csv
"""

import argparse
import sys
import pandas as pd
import numpy as np

# Commission-based tier ceilings (UGX) — ordered highest to lowest
COMMISSION_TIERS = {
    "Diamond":         1_000_000,
    "Titanium":          750_000,
    "Platinum":          500_000,
    "Gold":              350_000,
    "Silver":            250_000,
    "Bronze":            100_000,
    "New Bronze":         50_000,
    "Below Threshold":         0,
}

# Ordered list for tier navigation (up/down one step)
TIER_ORDER = list(COMMISSION_TIERS.keys())  # Diamond first, Below Threshold last


def tier_ceiling(tier_name: str) -> int:
    return COMMISSION_TIERS.get(tier_name, 0)


def tier_one_below(tier_name: str) -> str:
    """Return the tier one step below (more conservative) than tier_name."""
    idx = TIER_ORDER.index(tier_name) if tier_name in TIER_ORDER else len(TIER_ORDER) - 1
    return TIER_ORDER[min(idx + 1, len(TIER_ORDER) - 1)]


def _clamp_to_one_step(base_tier: str, cap_tier: str) -> str:
    """
    Limit capacity_tier adjustment to ±1 step from the commission base tier.

    TIER_ORDER runs Diamond(0) … Below Threshold(7); lower index = higher ceiling.
    A Bronze base with Diamond capacity → Silver (one step up, not full Diamond).
    A Silver base with New Bronze capacity → Bronze (one step down, not New Bronze).
    """
    if base_tier not in TIER_ORDER or cap_tier not in TIER_ORDER:
        return base_tier
    base_idx = TIER_ORDER.index(base_tier)
    cap_idx  = TIER_ORDER.index(cap_tier)
    if cap_idx < base_idx:        # capacity_tier is more generous → one step up
        adj_idx = base_idx - 1
    elif cap_idx > base_idx:      # capacity_tier is more conservative → one step down
        adj_idx = base_idx + 1
    else:
        adj_idx = base_idx        # same tier — no change
    return TIER_ORDER[max(0, min(adj_idx, len(TIER_ORDER) - 1))]


def compute_option_b_ceiling(row) -> float:
    """
    Option B ceiling logic (bounded adjustment):

      1. Commission eligibility gate — if commission < 50,000 (Below Threshold),
         return 0 immediately. capacity_tier cannot override eligibility.
      2. capacity_tier adjusts the ceiling ±1 tier from the commission base.
         A Bronze commission agent with Diamond behaviour reaches Silver, not Diamond.
      3. Anomaly penalty — cap at one tier below commission base, floor at New Bronze.
         Unusual behaviour reduces limits but never eliminates eligibility.
      4. Thin-file cap still applies (same as Option A).
    """
    base_tier  = row.get("agent_category", "Below Threshold")
    cap_tier   = row.get("capacity_tier",  None)
    is_anomaly = bool(row.get("is_anomaly", 0))
    thin_file  = bool(row.get("thin_file_flag", 0))

    # Step 1: base ceiling from commission tier
    base_ceiling = tier_ceiling(base_tier)

    # Eligibility gate: commission must clear the minimum threshold.
    # capacity_tier is an adjustment signal, not an eligibility override.
    if base_ceiling == 0:
        return 0.0

    # Step 2: adjust ±1 tier from commission base using capacity_tier
    if pd.notna(cap_tier) and cap_tier in COMMISSION_TIERS:
        adjusted_tier    = _clamp_to_one_step(base_tier, cap_tier)
        adjusted_ceiling = tier_ceiling(adjusted_tier)
    else:
        adjusted_ceiling = base_ceiling

    # Step 3: anomaly penalty — one tier below commission base, floor at New Bronze
    if is_anomaly:
        penalty_tier = tier_one_below(base_tier)
        # Anomaly reduces limits but never eliminates eligibility for commissioned agents
        if penalty_tier == "Below Threshold":
            penalty_tier = "New Bronze"
        penalty_ceiling  = tier_ceiling(penalty_tier)
        adjusted_ceiling = min(adjusted_ceiling, penalty_ceiling)

    # Thin-file cap still applies (same as Option A)
    if thin_file:
        adjusted_ceiling = min(adjusted_ceiling, 100_000)

    return float(adjusted_ceiling)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--input",  default="output/engine_test_output.csv")
    p.add_argument("--output", default="output/options_comparison.csv")
    args = p.parse_args(argv)

    print(f"\nLoading: {args.input}")
    try:
        df = pd.read_csv(args.input)
    except FileNotFoundError:
        print(f"ERROR: file not found — {args.input}")
        sys.exit(1)

    print(f"Agents loaded: {len(df):,}\n")

    # Option A limit = assigned_limit from current run
    df["option_a_limit"]    = df["assigned_limit"]
    df["option_a_category"] = df.get("agent_category", "Unknown")

    # Check required columns
    has_capacity_tier = "capacity_tier" in df.columns
    has_anomaly       = "is_anomaly"    in df.columns
    has_thin_file     = "thin_file_flag" in df.columns

    if not has_capacity_tier:
        print("WARNING: capacity_tier column not in output — Option B will equal Option A.")
        print("         Re-run run.bat to get the latest output with segmentation columns.\n")

    # Option B limit
    df["option_b_ceiling"] = df.apply(compute_option_b_ceiling, axis=1)
    # Scale option_b_limit proportionally: keep same ratio of assigned/ceiling as Option A
    # so the other caps (usage, prior exposure, PD risk) are still respected.
    a_ceiling = df["agent_tier_ceiling_multiplier"] * 1_000_000 if "agent_tier_ceiling_multiplier" in df.columns else df["option_a_limit"]
    ratio = np.where(a_ceiling > 0, df["option_a_limit"] / a_ceiling.clip(lower=1), 1.0)
    df["option_b_limit"] = (df["option_b_ceiling"] * ratio).clip(lower=0).round(-2)

    # Movement classification
    df["limit_change"]    = df["option_b_limit"] - df["option_a_limit"]
    df["movement"]        = pd.cut(
        df["limit_change"],
        bins=[-float("inf"), -1, 1, float("inf")],
        labels=["Down (more conservative)", "No change", "Up (higher limit)"],
    )

    # ── Summary ──────────────────────────────────────────────────────────────
    print("=" * 65)
    print(" OPTION A  (commission → hard ceiling)")
    print("=" * 65)
    if "agent_category" in df.columns:
        print("\nagent_category distribution:")
        print(df["agent_category"].value_counts().to_string())
    print(f"\nLimit stats (UGX):")
    print(df["option_a_limit"].describe().apply(lambda x: f"{x:,.0f}").to_string())

    print("\n" + "=" * 65)
    print(" OPTION B  (commission base + behavioural adjustment)")
    print("=" * 65)
    if has_capacity_tier:
        print("\ncapacity_tier distribution (segmentation):")
        print(df["capacity_tier"].value_counts().to_string())
    if has_anomaly:
        print(f"\nAnomaly-flagged agents: {int(df['is_anomaly'].sum()):,} "
              f"({df['is_anomaly'].mean()*100:.1f}%)")
    print(f"\nLimit stats (UGX):")
    print(df["option_b_limit"].describe().apply(lambda x: f"{x:,.0f}").to_string())

    print("\n" + "=" * 65)
    print(" MOVEMENT  (Option B vs Option A)")
    print("=" * 65)
    print(df["movement"].value_counts().to_string())
    up   = df[df["limit_change"] > 0]
    down = df[df["limit_change"] < 0]
    print(f"\nAgents moving UP  : {len(up):,}  | avg increase: {up['limit_change'].mean():,.0f} UGX")
    print(f"Agents moving DOWN: {len(down):,}  | avg decrease: {down['limit_change'].mean():,.0f} UGX")
    print(f"Total float exposure change: {df['limit_change'].sum():,.0f} UGX")

    if has_anomaly:
        anomaly_df = df[df["is_anomaly"] == 1]
        print(f"\nAnomaly agents — Option A mean: {anomaly_df['option_a_limit'].mean():,.0f} "
              f"| Option B mean: {anomaly_df['option_b_limit'].mean():,.0f}")

    if "agent_category" in df.columns:
        tier_summary = (
            df.groupby("agent_category")
            .agg(
                n=("msisdn", "count"),
                opt_a_mean=("option_a_limit", "mean"),
                opt_b_mean=("option_b_limit", "mean"),
            )
            .assign(
                change_pct=lambda t: (
                    (t["opt_b_mean"] - t["opt_a_mean"])
                    / t["opt_a_mean"].clip(lower=1) * 100
                ).round(1)
            )
        )
        tier_summary["opt_a_mean"] = tier_summary["opt_a_mean"].round(0).astype(int)
        tier_summary["opt_b_mean"] = tier_summary["opt_b_mean"].round(0).astype(int)
        print("\n" + "=" * 65)
        print(" BY COMMISSION TIER  (agent_category)")
        print("=" * 65)
        print(tier_summary.to_string())

    # ── Output columns ────────────────────────────────────────────────────────
    keep_cols = [
        "msisdn", "commission" if "commission" in df.columns else None,
        "agent_category", "capacity_tier" if has_capacity_tier else None,
        "is_anomaly" if has_anomaly else None,
        "thin_file_flag" if has_thin_file else None,
        "option_a_limit", "option_b_limit", "limit_change", "movement",
        "cal_pd", "risk_tier", "score_source",
    ]
    keep_cols = [c for c in keep_cols if c and c in df.columns]

    out = df[keep_cols].copy()
    out.to_csv(args.output, index=False)
    print(f"\nComparison written to: {args.output}")
    print(f"Columns: {keep_cols}\n")


if __name__ == "__main__":
    main()
