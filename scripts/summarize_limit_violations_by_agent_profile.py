"""
Breaks down check_disbursements_vs_assigned_limit.py's per-transaction
export by instruct_to_fro_user_prf -- the recipient agent's own profile
field as recorded natively in fin_log, independent of anything the PD
engine computed.

This checks a different hypothesis than summarize_limit_violations_by_risk.py:
rather than asking whether violations concentrate among the PD model's
riskiest agents, this asks whether real disbursement behavior instead lines
up with (or is actually governed by) this pre-existing, native profile
classification -- which would suggest the PD engine's assigned_limit isn't
the control actually driving disbursement decisions at all.

Also cross-tabs agent_profile against risk_tier (where both are present) to
see whether the native profile and the model's risk tier agree with each
other in the first place.

Usage:
    python scripts\\summarize_limit_violations_by_agent_profile.py ^
        --file disbursements_vs_limit_per_transaction.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

PROFILE_COL = "instruct_to_fro_user_prf"


def _summarize(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    g = df.groupby(group_col, observed=True)
    return pd.DataFrame({
        "n_transactions": g.size(),
        "violation_rate": g["exceeds_limit"].mean(),
        "n_violations": g["exceeds_limit"].sum(),
        "median_pct_of_limit": g["pct_of_limit"].median(),
        "median_excess_when_violating": df[df["exceeds_limit"]].groupby(group_col, observed=True)["excess_amount"].median(),
    })


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default="disbursements_vs_limit_per_transaction.csv")
    args = ap.parse_args()

    path = Path(args.file)
    if not path.exists():
        sys.exit(f"ERROR: file not found: {path}")

    df = pd.read_csv(path)
    if PROFILE_COL not in df.columns:
        sys.exit(
            f"ERROR: '{PROFILE_COL}' not in {path}. Re-run "
            f"check_disbursements_vs_assigned_limit.py against a disbursements file that "
            f"includes this column, then re-run this script."
        )
    df["exceeds_limit"] = df["exceeds_limit"].astype(bool)
    print(f"Rows: {len(df):,}\n")

    print("=" * 78)
    print(f"Violation rate by {PROFILE_COL} (native agent profile, independent of the PD model)")
    print("=" * 78)
    tbl = _summarize(df, PROFILE_COL).sort_values("n_transactions", ascending=False)
    print(tbl.round(4).to_string())

    if "risk_tier" in df.columns:
        print(f"\n{'=' * 78}")
        print(f"Cross-tab: {PROFILE_COL} vs. risk_tier (agent counts, distinct msisdn if available)")
        print("=" * 78)
        key_cols = [c for c in ["msisdn", PROFILE_COL, "risk_tier"] if c in df.columns]
        agents = df[key_cols].drop_duplicates()
        ct = pd.crosstab(agents[PROFILE_COL], agents["risk_tier"])
        print(ct.to_string())
        print(
            "\nIf each native profile maps cleanly onto one risk_tier, the two "
            "classifications broadly agree. A scattered cross-tab means the native "
            "profile and the PD model's risk assessment disagree about who's risky."
        )

    print(
        "\nInterpretation: if violation_rate varies sharply by native profile in a way "
        "that does NOT track risk_tier (see summarize_limit_violations_by_risk.py), "
        "disbursement behavior may be keyed off this pre-existing profile system rather "
        "than the PD engine's assigned_limit -- i.e. the model's limit may not be the "
        "control actually governing real disbursement decisions."
    )


if __name__ == "__main__":
    main()
