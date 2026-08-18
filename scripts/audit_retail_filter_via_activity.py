"""
Second, independent validation for the profile-name-based retail filter
(apply_retail_agent_filter.py) -- using customers_served_3m behavior at
the PROFILE-AGGREGATE level, not the individual-agent level.

Why aggregate, not per-agent: apply_retail_agent_filter.py deliberately
never excludes an individual agent for being zero-activity, since 35.5%
of the core, unambiguously-retail "Agent Silver Class" population is
also zero-activity in the last 3 months (dormant/new agents, not
non-retail ones) -- a per-agent activity cutoff would wrongly strip a
third of the real retail book. But a PROFILE TYPE where nearly every
agent is zero-activity (e.g. "NO END USER TRANSACTIONS ALLOWED" showed
exactly 100% zero, confirmed earlier this session) is a structural
signal about the account type itself, not about any one agent's
dormancy -- that's a safe, aggregate-level signal to audit the
name-based classification against.

Why this matters: profile NAMES can change upstream (renamed, versioned,
a new profile type introduced) and a static string list silently falls
back to "unclassified -> excluded" for anything it doesn't recognize --
safe, but requires a human to notice and update the list. This script
gives that human evidence to act on: for every profile (known or brand
new), its aggregate zero-customer rate, and an explicit flag wherever
that behavior disagrees with the current name-based classification.

Usage:
    python audit_retail_filter_via_activity.py ^
        --transaction-file data\\mfs_daily_agent_mart_20260731.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from apply_retail_agent_filter import classify_agent_profile, _normalize_profile_text  # noqa: E402

# Profiles where "REVIEW: classified {rebalancer/derisk} but shows real
# customer activity" is expected, permanent behavior, already confirmed
# with business -- not a new finding each run. Excluded from n_review (so
# the exit-code gate a .bat file checks isn't perpetually tripped by the
# same known cases) but still shown in the table with their own flag, so
# they stay visible rather than silently disappearing.
KNOWN_ACCEPTED_DISAGREEMENTS = {
    "agent derisk class",
    "super agent branch account",
    "super agent master account",
    "master agent silver class",
    "master agent bronze class",
}

# Unclassified profiles with a genuinely ambiguous behavioral signal
# (real but modest activity, well below the confirmed-retail baseline),
# deliberately left unclassified pending a business decision rather than
# guessed at -- population is tiny enough (189 agents combined, 0.04% of
# the file) that it's not worth blocking the review gate on. Still
# excluded by default via the unclassified safety net either way; this
# only silences the recurring flag. Move to RETAIL_PROFILES or
# NON_RETAIL_PROFILES in apply_retail_agent_filter.py once business
# confirms, and remove from here.
KNOWN_PENDING_UNCLASSIFIED = {
    "mobile money dsd agent profile",
    "merchant master agent hierarchy account",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--transaction-file", default="data/mfs_daily_agent_mart_20260731.csv")
    ap.add_argument("--profile-col", default="agent_profile")
    ap.add_argument("--zero-rate-threshold", type=float, default=0.90,
                     help="Profile-level pct_zero above which a profile is flagged behaviorally non-retail")
    ap.add_argument("--min-profile-n", type=int, default=20,
                     help="Minimum agents in a profile for its behavioral signal to be trusted")
    ap.add_argument("--out", default="retail_filter_activity_audit.csv")
    args = ap.parse_args()

    txn_path = Path(args.transaction_file)
    if not txn_path.exists():
        sys.exit(f"ERROR: transaction file not found: {txn_path}")

    df = pd.read_csv(txn_path, sep=",", encoding="utf-8-sig")
    if args.profile_col not in df.columns:
        sys.exit(f"ERROR: '{args.profile_col}' column not found. Found: {list(df.columns)[:30]}")

    cust_parts = ["cash_out_cust_3m", "cash_in_cust_3m", "voucher_cust_3m", "payment_cust_3m"]
    present_cust_parts = [c for c in cust_parts if c in df.columns]
    if not present_cust_parts:
        cust_like = [c for c in df.columns if "cust" in c.lower()]
        sys.exit(
            f"ERROR: none of the expected *_cust_3m columns found.\n"
            f"Columns containing 'cust': {cust_like}"
        )
    df["customers_served_3m"] = sum(
        pd.to_numeric(df[c], errors="coerce").fillna(0) for c in present_cust_parts
    )

    # Static classification: what the name-based filter currently says,
    # independent of --include-rebalancers/--include-derisk (this audit
    # cares about the CATEGORY, not the current inclusion decision).
    df["_retail_category"] = df[args.profile_col].apply(
        lambda p: classify_agent_profile(p, include_rebalancers=False, include_derisk=False)
    )

    profile_tbl = (
        df.groupby(args.profile_col)
        .agg(
            n=("customers_served_3m", "size"),
            median_customers_served_3m=("customers_served_3m", "median"),
            mean_customers_served_3m=("customers_served_3m", "mean"),
            pct_zero=("customers_served_3m", lambda s: float((s == 0).mean())),
            static_category=("_retail_category", "first"),
        )
        .reset_index()
    )

    def _behavioral_signal(row):
        if row["n"] < args.min_profile_n:
            return "insufficient_data"
        return "mostly_zero" if row["pct_zero"] >= args.zero_rate_threshold else "has_real_activity"

    profile_tbl["behavioral_signal"] = profile_tbl.apply(_behavioral_signal, axis=1)

    def _flag(row):
        cat, sig = row["static_category"], row["behavioral_signal"]
        if sig == "insufficient_data":
            return ""
        if cat == "unclassified":
            key = _normalize_profile_text(row[args.profile_col])
            if key in KNOWN_PENDING_UNCLASSIFIED:
                return "acknowledged (pending): ambiguous, tiny population, awaiting business confirmation"
            return (
                "SUGGEST non-retail (behaves like the confirmed non-retail profiles)"
                if sig == "mostly_zero"
                else "SUGGEST retail-like -- confirm with business before adding to RETAIL_PROFILES"
            )
        if cat == "retail" and sig == "mostly_zero":
            return "REVIEW: classified retail but behaves non-customer-facing"
        if cat in ("non_retail", "rebalancer", "derisk") and sig == "has_real_activity":
            key = _normalize_profile_text(row[args.profile_col])
            if key in KNOWN_ACCEPTED_DISAGREEMENTS:
                return f"acknowledged (expected): classified {cat}, known to show some real activity"
            return f"REVIEW: classified {cat} but shows real customer activity"
        return "consistent"

    profile_tbl["audit_flag"] = profile_tbl.apply(_flag, axis=1)

    def _flag_priority(v: str) -> int:
        if v.startswith("REVIEW") or v.startswith("SUGGEST"):
            return 0
        if v.startswith("acknowledged"):
            return 1
        return 2  # "" or "consistent"

    profile_tbl["_flag_priority"] = profile_tbl["audit_flag"].apply(_flag_priority)
    profile_tbl = profile_tbl.sort_values(
        by=["_flag_priority", "n"], ascending=[True, False]
    ).drop(columns=["_flag_priority"])

    print(f"Total agents: {len(df):,}  |  distinct profiles: {len(profile_tbl):,}\n")
    print(f"Threshold: profile-level pct_zero >= {args.zero_rate_threshold:.0%} => 'mostly_zero'; "
          f"profiles with fewer than {args.min_profile_n} agents are not judged.\n")
    print(profile_tbl.to_string(index=False))

    n_suggest = int(profile_tbl["audit_flag"].str.startswith("SUGGEST", na=False).sum())
    n_disagree = int(profile_tbl["audit_flag"].str.startswith("REVIEW", na=False).sum())
    n_acknowledged = int(profile_tbl["audit_flag"].str.startswith("acknowledged", na=False).sum())
    # Gate on genuinely new findings only -- "acknowledged" disagreements
    # are expected, permanent, already-confirmed business behavior, so
    # they don't count toward n_review (the exit-code-2 gate below).
    n_review = n_suggest + n_disagree
    print(f"\n{n_review} profile(s) need review: {n_disagree} disagree with the current "
          f"classification, {n_suggest} are unclassified with a behavioral suggestion "
          f"(worth confirming with business, then adding to the appropriate list in "
          f"apply_retail_agent_filter.py).")
    if n_acknowledged:
        print(f"{n_acknowledged} additional profile(s) show a known, already-acknowledged "
              f"disagreement (see 'acknowledged (expected)' rows above) -- not counted "
              f"toward the review gate.")

    profile_tbl.to_csv(args.out, index=False)
    print(f"\nFull per-profile audit written to: {args.out}")

    # Exit code lets a .bat file chain into apply_retail_agent_filter.py
    # only when there's nothing to review: 0 = clean, 2 = flags found
    # (distinct from 1, which argparse/sys.exit(str) already use for a
    # genuine script error like a missing file).
    if n_review > 0:
        print(f"\nExiting with code 2 ({n_review} flag(s) found) -- review before proceeding.")
        sys.exit(2)


if __name__ == "__main__":
    main()
