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
    # right=False: bins are [a, b), matching the documented intent exactly
    # ("R_i >= 1.10" and "0.90 <= R_i < 1.10"). pd.cut's default (right=True,
    # bins (a, b]) would put R==0.90 in the BELOW band instead of AT, and
    # R==1.10 in the AT band instead of ABOVE -- found by a boundary-value
    # test fixture (R=0.90 exactly), not observed in the real data pulled
    # so far (an exact-boundary float match is highly unlikely there), but
    # a real latent correctness bug regardless.
    return pd.cut(pd.to_numeric(r, errors="coerce"), bins=R_BAND_EDGES, labels=R_BAND_LABELS, right=False)


def attach_risk_tier(df: pd.DataFrame, engine_output_file) -> pd.DataFrame:
    """risk_tier is not a Decision Log column (by frozen schema design) --
    read from the same shadow run's own engine output CSV and joined in
    for this detail dump only, same convention as
    check_stage_4c_shadow_distribution_sanity.py's load_joined()."""
    eng = pd.read_csv(engine_output_file, usecols=lambda c: c in {"msisdn", "risk_tier"})
    if "risk_tier" not in eng.columns:
        raise ValueError(f"{engine_output_file} has no risk_tier column.")
    eng["_key"] = _norm_msisdn(eng["msisdn"])
    eng = eng.drop_duplicates(subset=["_key"])
    return df.merge(eng[["_key", "risk_tier"]], on="_key", how="left")


# Column order for the --severity-detail-band row dump, per the reviewer's
# exact request: a narrow decomposition, not a new analysis. challenger_risk_signal
# is included alongside cal_pd (both already distinct Decision Log columns --
# cal_pd is NULL on the thin-file route, challenger_risk_signal is the pooled
# value regardless of route; both are shown rather than picking one).
SEVERITY_DETAIL_COLUMNS = [
    "agent_msisdn", "max_pct_of_live_disbursed", "n_months_over_threshold",
    "float_activity_value_1m", "commission", "capacity_estimable",
    "capacity_shadow_k0_05", "challenger_risk_signal", "cal_pd",
    "shadow_multiplier_base", "l_current", "l_full_shadow_k0_05_base",
    "disagreement_ratio_shadow_base", "r_band", "risk_tier",
]


def severity_band_detail(diamond_a_rows: pd.DataFrame, severity_label: str) -> pd.DataFrame:
    """Row-level dump of exactly SEVERITY_DETAIL_COLUMNS for one severity
    band -- no aggregation, no new thresholds. Sorted by r_band (BELOW
    first) then by disagreement_ratio_shadow_base ascending within each
    band, so the printed table itself reads as a worst-to-best ordering."""
    sub = diamond_a_rows.copy()
    sub["severity_band"] = pd.cut(
        sub["max_pct_of_live_disbursed"], bins=SEVERITY_EDGES, labels=SEVERITY_LABELS, right=False
    )
    band = sub[sub["severity_band"].astype(str) == severity_label].copy()
    cols = [c for c in SEVERITY_DETAIL_COLUMNS if c in band.columns]
    band["_r_band_order"] = band["r_band"].map({label: i for i, label in enumerate(R_BAND_LABELS)})
    band = band.sort_values(
        ["_r_band_order", "disagreement_ratio_shadow_base"], ascending=[True, True], na_position="first"
    )
    return band[cols]


def build_joined(decision_log_df: pd.DataFrame, diamond_a_df: pd.DataFrame) -> pd.DataFrame:
    """Left join from the Decision Log's full scored population -- so the
    match rate against the Diamond A file is itself visible (agents in a
    different population snapshot/run than this shadow run are expected
    to not all match), rather than silently restricting to only matched
    rows from the start. diamond_a_df is the FULL population returned by
    load_diamond_a_population() (every agent with ANY disbursement
    record, is_diamond_a True or False) -- NOT pre-filtered to
    is_diamond_a==True -- so has_disbursement_record can distinguish
    "non-Diamond-A but has a disbursement history" from "no disbursement
    record at all" (the latter must never be silently folded into the
    former)."""
    joined = decision_log_df.merge(
        diamond_a_df[["_key", "is_diamond_a", "max_pct_of_live_disbursed", "n_months_over_threshold"]],
        on="_key", how="left",
    )
    joined["has_disbursement_record"] = joined["_key"].isin(diamond_a_df["_key"])
    # .astype(bool) is NOT cosmetic: the merge leaves is_diamond_a as
    # object dtype (mixed real bool / NaN before fillna), and `~` on an
    # object-dtype Series of Python True/False does raw bitwise complement
    # (~True == -2, ~False == -1 -- BOTH truthy), silently selecting every
    # row instead of negating. Found via a real test failure where the
    # "non-Diamond-A" filter returned the entire population, Diamond A
    # agents included.
    joined["is_diamond_a"] = joined["is_diamond_a"].fillna(False).astype(bool)
    joined["r_band"] = classify_r_band(joined["disagreement_ratio_shadow_base"])
    return joined


def compute_r_capacity(df: pd.DataFrame) -> pd.Series:
    """R_capacity = capacity_shadow_k0_05 / l_current -- the SAME guard
    convention already used for disagreement_ratio_shadow_base (l_current
    must be non-null and non-zero, else NaN, never Inf). Never recomputes
    capacity_shadow_k0_05 itself -- reads it straight from the Decision
    Log, same discipline as R_full (disagreement_ratio_shadow_base)."""
    cap = pd.to_numeric(df["capacity_shadow_k0_05"], errors="coerce")
    lc = pd.to_numeric(df["l_current"], errors="coerce")
    return pd.Series(np.where(lc.notna() & (lc != 0), cap / lc, np.nan), index=df.index)


def decomposition_crosstab(df: pd.DataFrame, population_label: str) -> pd.DataFrame:
    """Answers: is a population's BELOW-current result coming from
    Capacity(F) itself, or from the subsequent C3 risk adjustment? Cross-
    tabs R_capacity vs. R_full at the pivot of 1.0 (matching the
    reviewer's literal 2x2 table -- NOT the 0.90/1.10 'agreement band'
    used elsewhere in this script, a deliberately different, simpler
    question). The capacity<1 & full>=1 cell is mechanically impossible
    while shadow_multiplier_base <= 1 always holds (R_full = R_capacity *
    M_C3 <= R_capacity) -- reported and flagged loudly if ever nonzero,
    never silently dropped."""
    estimable = df[df["capacity_estimable"] == True].copy()  # noqa: E712
    not_estimable_n = int((df["capacity_estimable"] == False).sum())  # noqa: E712
    r_cap = compute_r_capacity(estimable)
    r_full = pd.to_numeric(estimable["disagreement_ratio_shadow_base"], errors="coerce")
    valid = r_cap.notna() & r_full.notna()
    r_cap, r_full = r_cap[valid], r_full[valid]
    n = len(r_cap)

    cap_below = r_cap < 1.0
    full_below = r_full < 1.0

    rows = [
        {"population": population_label, "cell": "capacity>=1 & full>=1 (FULL_ABOVE_CURRENT)",
         "n": int((~cap_below & ~full_below).sum())},
        {"population": population_label,
         "cell": "capacity>=1 & full<1 (CAPACITY_ABOVE_RISK_REDUCED_BELOW_CURRENT)",
         "n": int((~cap_below & full_below).sum())},
        {"population": population_label, "cell": "capacity<1 & full<1 (CAPACITY_BELOW_CURRENT)",
         "n": int((cap_below & full_below).sum())},
        {"population": population_label,
         "cell": "capacity<1 & full>=1 (MECHANICALLY IMPOSSIBLE -- flag if nonzero)",
         "n": int((cap_below & ~full_below).sum())},
        {"population": population_label, "cell": "pct_capacity_below_1.0 (Capacity(F) itself, pre-C3)",
         "n": np.nan, "pct": round(float(cap_below.mean() * 100), 2) if n else np.nan},
        {"population": population_label, "cell": "pct_full_below_1.0 (after C3)",
         "n": np.nan, "pct": round(float(full_below.mean() * 100), 2) if n else np.nan},
        {"population": population_label, "cell": "(capacity_estimable=False, excluded)", "n": not_estimable_n},
        {"population": population_label, "cell": "n_valid (capacity_estimable=True, both ratios defined)", "n": n},
    ]
    out = pd.DataFrame(rows)
    if "pct" not in out.columns:
        out["pct"] = np.nan
    if n:
        for i in range(4):
            out.loc[i, "pct"] = round(out.loc[i, "n"] / n * 100, 2)
    return out


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


def compute_lift(treatment_summary: pd.DataFrame, reference_summary: pd.DataFrame, band: str) -> dict:
    """Is the treatment group (Diamond A) OVER-represented in a given
    R-band relative to some reference population? lift > 1 means yes.
    Reports BOTH lift (ratio) and the percentage-point difference
    alongside it, per the reviewer's own point: lift can look large
    purely because the reference prevalence is low, while the pp
    difference shows the practical magnitude directly."""
    t_pct = treatment_summary.loc[treatment_summary["r_band"] == band, "pct"].iloc[0]
    r_pct = reference_summary.loc[reference_summary["r_band"] == band, "pct"].iloc[0]
    lift = t_pct / r_pct if r_pct else np.nan
    return {"r_band": band, "treatment_pct": t_pct, "reference_pct": r_pct,
            "lift": lift, "pp_difference": t_pct - r_pct}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--live-shadow-vs-category-file", required=True)
    p.add_argument("--decision-log-file", required=True)
    p.add_argument("--diamond-a-threshold", type=float, default=DIAMOND_A_THRESHOLD)
    p.add_argument("--out-prefix", default=None)
    p.add_argument("--severity-detail-band", default=None, choices=SEVERITY_LABELS,
                    help="If given, print+write a row-level decomposition (SEVERITY_DETAIL_COLUMNS "
                         "only, no aggregation) for Diamond A agents in exactly this severity band. "
                         "Requires --engine-output-file for risk_tier.")
    p.add_argument("--engine-output-file", default=None,
                    help="Same shadow run's engine output CSV, for risk_tier (not a Decision Log "
                         "column). Required only when --severity-detail-band is given.")
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
    n_non_diamond_a_in_file = int((~diamond_a["is_diamond_a"]).sum())
    print(f"\nAgents with any disbursement record in {args.live_shadow_vs_category_file}: {len(diamond_a):,}")
    print(f"  Diamond A (>= {args.diamond_a_threshold:.0%} in at least one month): {n_diamond_a_agents:,}")
    print(f"  Non-Diamond-A (same file, same eligibility universe): {n_non_diamond_a_in_file:,}")
    arithmetic_ok = n_diamond_a_agents + n_non_diamond_a_in_file == len(diamond_a)
    print(f"  Arithmetic check: {n_diamond_a_agents:,} + {n_non_diamond_a_in_file:,} "
          f"{'==' if arithmetic_ok else '!='} {len(diamond_a):,} total "
          f"-- {'OK' if arithmetic_ok else 'MISMATCH -- investigate before interpreting anything below'}")

    joined = build_joined(dlog, diamond_a)
    n_matched_diamond_a = int(joined["is_diamond_a"].sum())
    n_matched_non_diamond_a = int((joined["has_disbursement_record"] & ~joined["is_diamond_a"]).sum())
    print(f"\nDecision Log rows (this shadow run): {len(joined):,}")
    print(f"  Matched to Diamond A: {n_matched_diamond_a:,} ({n_matched_diamond_a / n_diamond_a_agents * 100:.2f}% "
          f"of all Diamond A agents in the file)")
    print(f"  Matched to non-Diamond-A disbursed: {n_matched_non_diamond_a:,} "
          f"({n_matched_non_diamond_a / n_non_diamond_a_in_file * 100:.2f}% of all non-Diamond-A disbursed "
          f"agents in the file)")
    print("  (match rate < 100% on either side means this shadow run's scored population differs from "
          "the file the Diamond A list came from -- e.g. a retail-filtered run vs. a full-population file)")

    # build_joined() already attached max_pct_of_live_disbursed/
    # n_months_over_threshold via its own merge -- re-merging the same
    # columns here would just produce pandas-suffixed duplicates.
    diamond_a_rows = joined[joined["is_diamond_a"]].copy()
    non_diamond_a_disbursed_rows = joined[joined["has_disbursement_record"] & ~joined["is_diamond_a"]].copy()
    overall_rows = joined  # includes Diamond A rows too, by design -- "overall" = the full scored population

    diamond_a_summary = summarize_r_band_distribution(diamond_a_rows, "Diamond A")
    non_diamond_a_summary = summarize_r_band_distribution(
        non_diamond_a_disbursed_rows, "Non-Diamond-A disbursed (direct complement)")
    overall_summary = summarize_r_band_distribution(overall_rows, "OVERALL (full scored population)")

    pd.set_option("display.width", 160)
    print("\n" + "=" * 80)
    print("R_i = L_full_shadow_k0_05_base / l_current -- band distribution")
    print("Diamond A vs. its direct complement (same disbursement-record population) vs. overall")
    print("=" * 80)
    print(pd.concat([diamond_a_summary, non_diamond_a_summary, overall_summary], ignore_index=True)
          .to_string(index=False))

    print("\n" + "=" * 80)
    print("LIFT + pp DIFFERENCE: Diamond A vs. its direct complement (same eligibility universe)")
    print("(lift > 1 / pp_difference > 0 => Diamond A is over-represented in that band;")
    print(" this is the cleanest answer, since both sides have a disbursement history)")
    print("=" * 80)
    for band in R_BAND_LABELS:
        lift = compute_lift(diamond_a_summary, non_diamond_a_summary, band)
        print(f"  {band:55s} Diamond A={lift['treatment_pct']:6.2f}%  "
              f"Non-Diamond-A disbursed={lift['reference_pct']:6.2f}%  "
              f"lift={lift['lift']:.2f}x  pp_diff={lift['pp_difference']:+.2f}")

    print("\n" + "=" * 80)
    print("LIFT + pp DIFFERENCE: Diamond A vs. the OVERALL scored population (original framing,")
    print("kept for continuity with the earlier result)")
    print("=" * 80)
    for band in R_BAND_LABELS:
        lift = compute_lift(diamond_a_summary, overall_summary, band)
        print(f"  {band:55s} Diamond A={lift['treatment_pct']:6.2f}%  "
              f"Overall={lift['reference_pct']:6.2f}%  lift={lift['lift']:.2f}x  "
              f"pp_diff={lift['pp_difference']:+.2f}")

    severity_summary = summarize_by_severity(diamond_a_rows)
    print("\n" + "=" * 80)
    print("Diamond A R-band distribution BY SEVERITY of historical over-disbursement")
    print("(descriptive sub-bucketing of the already->=200% Diamond A population --")
    print(" not a new business threshold)")
    print("=" * 80)
    print(severity_summary.to_string(index=False))

    print("\n" + "=" * 80)
    print("DECOMPOSITION: is the BELOW-current result coming from Capacity(F) itself,")
    print("or from the subsequent C3 risk adjustment? R_capacity = capacity_shadow_k0_05 /")
    print("l_current vs. R_full = l_full_shadow_k0_05_base / l_current (pivot at 1.0,")
    print("deliberately NOT the 0.90/1.10 agreement band used above -- a different question)")
    print("=" * 80)
    decomp_diamond_a = decomposition_crosstab(diamond_a_rows, "Diamond A")
    decomp_non_diamond_a = decomposition_crosstab(non_diamond_a_disbursed_rows, "Non-Diamond-A disbursed")
    decomp_overall = decomposition_crosstab(overall_rows, "OVERALL")
    print(pd.concat([decomp_diamond_a, decomp_non_diamond_a, decomp_overall], ignore_index=True)
          .to_string(index=False))

    if args.severity_detail_band:
        if not args.engine_output_file:
            print("\nERROR: --severity-detail-band requires --engine-output-file (for risk_tier).")
            return 1
        detail_rows = attach_risk_tier(diamond_a_rows, args.engine_output_file)
        detail = severity_band_detail(detail_rows, args.severity_detail_band)
        print("\n" + "=" * 80)
        print(f"SEVERITY DETAIL: Diamond A, severity={args.severity_detail_band} "
              f"({len(detail)} agents) -- row-level decomposition, no aggregation")
        print("=" * 80)
        print(detail.to_string(index=False))
        if args.out_prefix:
            detail.to_csv(f"{args.out_prefix}_severity_detail_{args.severity_detail_band.replace('%', 'pct')}.csv",
                           index=False)
            print(f"\nWrote {args.out_prefix}_severity_detail_"
                  f"{args.severity_detail_band.replace('%', 'pct')}.csv")

    if args.out_prefix:
        pd.concat([diamond_a_summary, non_diamond_a_summary, overall_summary], ignore_index=True).to_csv(
            f"{args.out_prefix}_r_band_summary.csv", index=False)
        severity_summary.to_csv(f"{args.out_prefix}_severity_breakdown.csv", index=False)
        diamond_a_rows.to_csv(f"{args.out_prefix}_diamond_a_matched_rows.csv", index=False)
        pd.concat([decomp_diamond_a, decomp_non_diamond_a, decomp_overall], ignore_index=True).to_csv(
            f"{args.out_prefix}_decomposition_crosstab.csv", index=False)
        print(f"\nWrote {args.out_prefix}_r_band_summary.csv / _severity_breakdown.csv / "
              f"_diamond_a_matched_rows.csv / _decomposition_crosstab.csv")

    return 0


if __name__ == "__main__":
    sys.exit(main())
