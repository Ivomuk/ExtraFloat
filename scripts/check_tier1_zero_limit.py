import pandas as pd

df = pd.read_csv("output/engine_test_output.csv")

if "combined_cap" not in df.columns:
    print(
        "combined_cap not in output/engine_test_output.csv -- re-run run.bat "
        "with --keep-intermediate first (same flag as before), then re-run "
        "this script."
    )
else:
    tier1 = df[df["risk_tier"] == "tier_1"].copy()
    print(f"tier_1 agents: {len(tier1):,}")

    tier1["assigned_limit_is_zero"] = tier1["assigned_limit"] == 0
    n_zero = int(tier1["assigned_limit_is_zero"].sum())
    print(f"  of which assigned_limit == 0: {n_zero:,} ({n_zero/len(tier1):.1%})")

    print("\n=== combined_cap within tier_1, split by whether assigned_limit == 0 ===")
    print(
        tier1.groupby("assigned_limit_is_zero")["combined_cap"]
        .agg(n="count", mean="mean", median="median", max="max")
        .to_string()
    )

    # Direct confirmation of the hypothesis: among the zero-assigned-limit
    # tier_1 agents, how many also have combined_cap == 0 (or near-zero)?
    zero_limit_tier1 = tier1[tier1["assigned_limit_is_zero"]]
    zero_cap_too = int((zero_limit_tier1["combined_cap"] <= 1.0).sum())
    print(
        f"\nOf the {len(zero_limit_tier1):,} tier_1 agents with assigned_limit == 0, "
        f"{zero_cap_too:,} ({zero_cap_too/max(1, len(zero_limit_tier1)):.1%}) "
        f"also have combined_cap <= 1 (i.e., effectively zero capacity)."
    )

    # If capacity-related intermediate columns are also present (only with
    # --keep-intermediate), show their shape too for the zero-limit tier_1
    # group specifically, to see which component is actually driving it.
    for col in ("capacity_cap", "recent_usage_cap", "prior_exposure_cap", "risk_cap"):
        if col in zero_limit_tier1.columns:
            print(f"\n{col} within zero-assigned-limit tier_1 agents:")
            print(zero_limit_tier1[col].describe())
