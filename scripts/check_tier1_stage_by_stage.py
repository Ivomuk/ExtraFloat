import pandas as pd

df = pd.read_csv("output/engine_test_output.csv")

stage_cols = [
    "capacity_cap", "recent_usage_cap", "prior_exposure_cap", "risk_cap",
    "is_thin_file", "prior_limit",
    "combined_cap_before_risk_guardrail",
    "combined_cap_after_risk_guardrail",
    "combined_cap_before_smoothing",
    "combined_cap",
    "assigned_limit",
]
present = [c for c in stage_cols if c in df.columns]
missing = [c for c in stage_cols if c not in df.columns]
if missing:
    print(f"Skipping missing columns: {missing}")

tier1_zero = df[(df["risk_tier"] == "tier_1") & (df["assigned_limit"] == 0)]
print(f"tier_1, assigned_limit==0 agents: {len(tier1_zero):,}\n")
print(tier1_zero[present].describe().to_string())

print("\n=== Same agents, first 10 rows raw (to see the actual per-agent progression) ===")
print(tier1_zero[present].head(10).to_string())
