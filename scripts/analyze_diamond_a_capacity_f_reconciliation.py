"""
analyze_diamond_a_capacity_f_reconciliation.py
=================================================
Diamond A x Capacity(F) reconciliation. Revisits the earlier,
observational "possible underfunding" question -- agents who
historically received disbursements >= 2x their engine's proposed
assigned_limit ("Diamond A", identified via
check_live_shadow_vs_category_limit.py's pct_of_live_disbursed column,
itself = disbursed_amount / assigned_limit) -- using the frozen,
INDEPENDENT Stage 4C Decision Log challenger (Capacity(F; k=0.05) x C3)
as a fresh signal, rather than re-using the old population's own
historical performance data (which is not re-derived here at all).

Question this answers: among Diamond A agents, what proportion does
the independent frozen challenger classify as having capacity ABOVE
vs. AT vs. BELOW their current live limit (R_i = L_full/L_current),
and is that proportion disproportionate relative to the overall
capacity-estimable population?

R_i is NEVER recomputed independently here -- it is read directly from
the Decision Log's own disagreement_ratio_shadow_base column
(R_i = L_full_shadow_k0_05_base / l_current by that column's own
definition), so this analysis can never drift from the frozen Decision
Log's own arithmetic.

Band convention: reuses this repo's already-established "90-110% =
agreement" threshold (first defined in
check_monthly_summary_vs_assigned_limit.py, restated independently in
check_live_shadow_vs_category_limit.py, restated independently again
here) rather than inventing a new one for R_i:
    R_i >= 1.10        -> "challenger supports MORE exposure"
    0.90 <= R_i < 1.10 -> "challenger broadly supports existing exposure"
    R_i < 0.90         -> "fundamentals do not support current exposure
                            under frozen Capacity(F)+C3"

Explicitly NOT causal (stated in every printed section, per the
reviewer's own framing): even a strong "Diamond A disproportionately in
the R>=1.10 group" result shows the shadow challenger INDEPENDENTLY
recommends more exposure for agents already observed at higher
disbursement -- it does not show that raising their limits is safe.
That remains the role of the eventual controlled-exposure/Tier B test.

Never touches k, C3, or the Capacity(F) formula. Purely observational
reconciliation of two already-frozen artifacts.

Usage:
    python scripts\\analyze_diamond_a_capacity_f_reconciliation.py ^
        --live-shadow-vs-category-file live_shadow_vs_category_limit.csv ^
        --decision-log-file output\\stage_4c_shadow\\stage_4c_decision_log_<run_id>.csv ^
        --out-prefix output\\stage_4c_shadow\\diamond_a_reconciliation
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# The "2x or more disbursement vs. the engine's proposed limit" criterion,
# stated by the reviewer, mapped onto check_live_shadow_vs_category_limit.py's
# existing pct_of_live_disbursed column (= disbursed_amount / assigned_limit) --
# not a new threshold, the same one already present in that script's own
# BUCKET_EDGES ("200%+ (far over)").
DIAMOND_A_THRESHOLD = 2.0

# Restated (not imported) from check_live_shadow_vs_category_limit.py /
# check_monthly_summary_vs_assigned_limit.py's shared "90-110% = agreement"
# convention -- same numeric edges, re-labeled for the R_i = L_full/L_current
# framing the reviewer specified (3 groups, not that script's 4).
R_BAND_EDGES = [-0.01, 0.90, 1.10, np.inf]
R_BAND_LABELS = [
    "BELOW: fundamentals do not support current exposure",
    "AT: challenger broadly supports existing exposure",
    "ABOVE: challenger supports MORE exposure",
]

# Purely descriptive sub-buckets of the already-over-2x Diamond A population
# itself (how far over 2x), so a stronger-vs-weaker R pattern among the most
# extreme over-disbursers is visible rather than averaged away. Not a new
# business threshold -- just finer bins within the already-defined cohort.
SEVERITY_EDGES = [2.0, 3.0, 5.0, 10.0, np.inf]
SEVERITY_LABELS = ["200-300%", "300-500%", "500-1000%", "1000%+"]


def _norm_msisdn(s: pd.Series) -> pd.Series:
    """Digit-only msisdn normalization -- restated (not imported) per the
    one-way scripts/ layering convention; matches
    check_live_shadow_vs_category_limit.py's own _normalize_msisdn."""
    out = s.astype(str).str.replace(r"[^0-9]", "", regex=True)
    return out.replace("", pd.NA)


def load_diamond_a_population(path, threshold: float = DIAMOND_A_THRESHOLD) -> pd.DataFrame:
    """One row per agent (not per agent-month): is_diamond_a = True if ANY
    of that agent's monthly rows had pct_of_live_disbursed >= threshold
    ("received 2x or more disbursement... " -- an ever-observed condition,
    not an average-across-months one). max_pct_of_live_disbursed is the
    single most extreme month, used for the severity sub-bucketing."""
    df = pd.read_csv(path)
    required = {"msisdn", "pct_of_live_disbursed"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} is missing required column(s) {missing} -- was it generated by "
            f"check_live_shadow_vs_category_limit.py WITH --monthly-summary-file (the "
            f"disbursement-enrichment columns, including pct_of_live_disbursed, are only "
            f"written when that optional join succeeds)? Columns present: {list(df.columns)}"
        )
    df["_key"] = _norm_msisdn(df["msisdn"])
    df["pct_of_live_disbursed"] = pd.to_numeric(df["pct_of_live_disbursed"], errors="coerce")
    valid = df.dropna(subset=["_key", "pct_of_live_disbursed"])

    grouped = valid.groupby("_key")["pct_of_live_disbursed"].agg(
        max_pct_of_live_disbursed="max",
        n_months_observed="count",
    )
    grouped["n_months_over_threshold"] = valid[valid["pct_of_live_disbursed"] >= threshold].groupby("_key").size()
    grouped["n_months_over_threshold"] = grouped["n_months_over_threshold"].fillna(0).astype(int)
    grouped["is_diamond_a"] = grouped["max_pct_of_live_disbursed"] >= threshold
    return grouped.reset_index()


def load_decision_log(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"agent_msisdn", "disagreement_ratio_shadow_base", "l_full_shadow_k0_05_base",
                "l_current", "capacity_estimable"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required Decision Log column(s): {missing}")
    df["_key"] = _norm_msisdn(df["agent_msisdn"])
    return df


def classify_r_band(r: pd.Series) -> pd.Series:
    return pd.cut(pd.to_numeric(r, errors="coerce"), bins=R_BAND_EDGES, labels=R_BAND_LABELS)


def build_joined(decision_log_df: pd.DataFrame, diamond_a_df: pd.DataFrame) -> pd.DataFrame:
    """Left join from the Decision Log's full scored population -- so the
    match rate against the Diamond A file is itself visible (agents in a
    different population snapshot/run than this shadow run are expected
    to not all match), rather than silently restricting to only matched
    rows from the start."""
    joined = decision_log_df.merge(
        diamond_a_df[["_key", "is_diamond_a", "max_pct_of_live_disbursed", "n_months_over_threshold"]],
        on="_key", how="left",
    )
    joined["is_diamond_a"] = joined["is_diamond_a"].fillna(False)
    joined["r_band"] = classify_r_band(joined["disagreement_ratio_shadow_base"])
    return joined


def summarize_r_band_distribution(df: pd.DataFrame, population_label: str) -> pd.DataFrame:
    """n / pct in each R-band, restricted to capacity_estimable rows (R is
    undefined otherwise) -- PLUS the capacity_estimable=False count
    reported alongside, never silently dropped out of the denominator
    without being shown."""
    estimable = df[df["capacity_estimable"] == True]  # noqa: E712
    not_estimable_n = int((df["capacity_estimable"] == False).sum())  # noqa: E712
    r_defined = estimable.dropna(subset=["r_band"])
    n_estimable_but_r_undefined = len(estimable) - len(r_defined)

    counts = r_defined["r_band"].value_counts().reindex(R_BAND_LABELS, fill_value=0)
    pct = (counts / len(r_defined) * 100).round(2) if len(r_defined) else counts.astype(float)

    rows = [{"population": population_label, "r_band": band, "n": int(counts[band]), "pct": float(pct[band])}
            for band in R_BAND_LABELS]
    rows.append({"population": population_label, "r_band": "(capacity_estimable=False, R undefined)",
                 "n": not_estimable_n, "pct": np.nan})
    rows.append({"population": population_label, "r_band": "(capacity_estimable=True but R still null)",
                 "n": n_estimable_but_r_undefined, "pct": np.nan})
    rows.append({"population": population_label, "r_band": "TOTAL", "n": len(df), "pct": np.nan})
    return pd.DataFrame(rows)


def summarize_by_severity(diamond_a_rows: pd.DataFrame) -> pd.DataFrame:
    """Within the Diamond A population only: does the R-band split shift
    for agents who were EVEN MORE extreme over-disbursers (300%, 500%,
    1000%+), rather than averaging all of Diamond A into one number."""
    sub = diamond_a_rows.copy()
    sub["severity_band"] = pd.cut(
        sub["max_pct_of_live_disbursed"], bins=SEVERITY_EDGES, labels=SEVERITY_LABELS, right=False
    )
    out = []
    for sev, grp in sub.groupby("severity_band", observed=True):
        tbl = summarize_r_band_distribution(grp, f"Diamond A, severity={sev}")
        out.append(tbl)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def compute_lift(diamond_a_summary: pd.DataFrame, overall_summary: pd.DataFrame, band: str) -> dict:
    """The single number that most directly answers the reviewer's
    question: is Diamond A OVER-represented in a given R-band relative to
    the overall capacity-estimable population? lift > 1 means yes."""
    d_pct = diamond_a_summary.loc[diamond_a_summary["r_band"] == band, "pct"].iloc[0]
    o_pct = overall_summary.loc[overall_summary["r_band"] == band, "pct"].iloc[0]
    lift = d_pct / o_pct if o_pct else np.nan
    return {"r_band": band, "diamond_a_pct": d_pct, "overall_pct": o_pct, "lift": lift}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--live-shadow-vs-category-file", required=True)
    p.add_argument("--decision-log-file", required=True)
    p.add_argument("--diamond-a-threshold", type=float, default=DIAMOND_A_THRESHOLD)
    p.add_argument("--out-prefix", default=None)
    args = p.parse_args(argv)

    print("=" * 80)
    print("Diamond A x Capacity(F) reconciliation")
    print("Observational only -- never touches k, C3, or the Capacity(F) formula.")
    print("Even a strong result here is NOT causal proof that raising Diamond A's")
    print("limits is safe -- it shows the independent challenger recommends more")
    print("exposure for this population, which is triangulating evidence toward an")
    print("eventual controlled-exposure/Tier B test, not a substitute for one.")
    print("=" * 80)

    diamond_a = load_diamond_a_population(args.live_shadow_vs_category_file, args.diamond_a_threshold)
    dlog = load_decision_log(args.decision_log_file)

    n_diamond_a_agents = int(diamond_a["is_diamond_a"].sum())
    print(f"\nDiamond A agents identified in {args.live_shadow_vs_category_file}: "
          f"{n_diamond_a_agents:,} (>= {args.diamond_a_threshold:.0%} of their own assigned_limit "
          f"in at least one month) out of {len(diamond_a):,} agents with any disbursement record.")

    joined = build_joined(dlog, diamond_a[diamond_a["is_diamond_a"]])
    n_matched = int(joined["is_diamond_a"].sum())
    print(f"Decision Log rows (this shadow run): {len(joined):,}")
    print(f"Of those, matched to a Diamond A agent_msisdn: {n_matched:,} "
          f"({n_matched / len(joined) * 100:.2f}% of this run's scored population; "
          f"{n_matched / n_diamond_a_agents * 100:.2f}% of all Diamond A agents found, if the "
          f"shadow run's population differs from the file the Diamond A list came from).")

    # build_joined() already attached max_pct_of_live_disbursed/
    # n_months_over_threshold via its own merge -- re-merging the same
    # columns here would just produce pandas-suffixed duplicates.
    diamond_a_rows = joined[joined["is_diamond_a"]].copy()
    overall_rows = joined  # includes Diamond A rows too, by design -- "overall" = the full scored population

    diamond_a_summary = summarize_r_band_distribution(diamond_a_rows, "Diamond A")
    overall_summary = summarize_r_band_distribution(overall_rows, "OVERALL (full scored population)")

    pd.set_option("display.width", 160)
    print("\n" + "=" * 80)
    print("R_i = L_full_shadow_k0_05_base / l_current -- band distribution")
    print("Diamond A vs. the overall scored population")
    print("=" * 80)
    print(pd.concat([diamond_a_summary, overall_summary], ignore_index=True).to_string(index=False))

    print("\n" + "=" * 80)
    print("LIFT: Diamond A's representation in each R-band vs. the overall population")
    print("(lift > 1 => Diamond A is over-represented in that band; this is the number")
    print(" that directly answers the question posed)")
    print("=" * 80)
    for band in R_BAND_LABELS:
        lift = compute_lift(diamond_a_summary, overall_summary, band)
        print(f"  {band:55s} Diamond A={lift['diamond_a_pct']:6.2f}%  "
              f"Overall={lift['overall_pct']:6.2f}%  lift={lift['lift']:.2f}x")

    severity_summary = summarize_by_severity(diamond_a_rows)
    print("\n" + "=" * 80)
    print("Diamond A R-band distribution BY SEVERITY of historical over-disbursement")
    print("(descriptive sub-bucketing of the already->=200% Diamond A population --")
    print(" not a new business threshold)")
    print("=" * 80)
    print(severity_summary.to_string(index=False))

    if args.out_prefix:
        pd.concat([diamond_a_summary, overall_summary], ignore_index=True).to_csv(
            f"{args.out_prefix}_r_band_summary.csv", index=False)
        severity_summary.to_csv(f"{args.out_prefix}_severity_breakdown.csv", index=False)
        diamond_a_rows.to_csv(f"{args.out_prefix}_diamond_a_matched_rows.csv", index=False)
        print(f"\nWrote {args.out_prefix}_r_band_summary.csv / _severity_breakdown.csv / "
              f"_diamond_a_matched_rows.csv")

    return 0


if __name__ == "__main__":
    sys.exit(main())
