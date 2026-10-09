"""
derive_capacity_frontier_from_business_state.py
===================================================
Analysis 4, Deliverable 1. Derives a FUNDAMENTALS-INDEXED, OUTCOME-SUPPORTED
EXPOSURE FRONTIER -- NOT Capacity(F) itself (see TERMINOLOGY below).

QUESTION: for agent-periods sharing a measured business-state band (float
or commission decile, kept independent), how far up the existing discrete
exposure-tier ladder does historical same-agent/same-anchor evidence
(Level 2's Table C2 pairing mechanic, restated here) support moving, before
subsequent performance deteriorates beyond an explicit tolerance?

TERMINOLOGY (load-bearing for governance -- say this precisely, every
docstring/column/banner in this script): this script derives a
HISTORICALLY-SUPPORTED EXPOSURE FRONTIER AT TOLERANCE TAU, estimated
jointly from F (fundamentals band), L (historical exposure tier), and Y
(subsequent outcome). It is NOT Capacity(F). Capacity(F) is reserved for a
later, separately-gated, FROZEN production mapping that -- once this
frontier is derived and reviewed -- takes only fundamentals as its runtime
input.

PRIOR ART, superseded here (not imported -- restated correctly, one-way
scripts/ layering convention): scripts/derive_capacity_risk_supported_
exposure_surface.py's derive_supported_frontier() implements the right
walk-forward SHAPE (walk exposure buckets ascending from one fixed,
sufficiently-populated baseline, comparing every candidate against that
SAME baseline, stopping at the first bucket that breaches an explicit
tolerance or lacks sufficient n). But it ran on capacity_research_
dataset.csv, whose forward-outcome columns Analysis 3's "Gate 0" finding
showed are computed over different, later loans than its own exposure
column (docs/analysis_3_findings.md), and it walked buckets CROSS-
SECTIONALLY across different agents within a (PD band, capacity tercile)
cell -- the same structure Analysis 3's Level 1 used, which Level 2/Table
C2 showed gives the OPPOSITE answer from the correct within-agent/same-
anchor read. This script fixes both: runs on loan_episode_capacity_
dataset.csv (Analysis 3's validated, correctly-time-aligned episode
dataset), and re-keys the walk to same-agent/same-anchor PAIRED evidence
(scripts/analyze_business_state_exposure_variation.py's build_table_c2
mechanic, restated), not cross-sectional buckets. No cal_pd/PD anywhere --
the episode dataset carries none at this grain, which is the correct
outcome per the architectural decision that capacity and risk stay
separate (that is C3/RiskMultiplier's job, not this script's).

BASELINE DESIGN: for each band, L0 = the LOWEST individual exposure tier
with marginal n_units_at_tier >= BASELINE_MIN_N -- a single, real,
homogeneous reference tier, never a pooled/blended one. (An earlier draft
pooled several low tiers into one blended baseline; this was rejected on
review because it manufactures an exposure-heterogeneous reference, and
Analysis 3's Table C/C2 showed absolute tier movement itself carries
information, so blending tiers into the reference weakens exactly the
estimand this frontier needs to preserve.) This makes deliberately NO
statement about tiers below L0 in that band -- preferred to a blended
reference just to speak about them.

EVIDENCE GATE: the decisive quantity for judging a candidate tier Lc is the
PAIRED count n_units_outcome_both for the SPECIFIC pair (L0, Lc) -- never
either tier's own marginal count. A tier can be common marginally and
still rarely co-occur with L0 under the same anchor.

CLASSIFICATION (per band, per tolerance tau, walking EXPOSURE_TIERS_UGX
ascending above L0):
  "reference_tier"              -- L0 itself. Qualifies on MARGINAL count;
                                    was never risk-tested against anything
                                    lower. Never called "supported" -- that
                                    label is earned only by a tested,
                                    passing candidate.
  "evidence_gap"                 -- n_units_outcome_both < EVIDENCE_GAP_MIN_N.
                                    Cannot be assessed. Does NOT halt the
                                    walk -- historical tier assignment is
                                    non-uniform, so a gap at one tier does
                                    not imply exposure above it is
                                    unsupported.
  "risk_breach"                  -- paired N adequate, delta_br_loan_weighted
                                    (pp, vs L0) > tolerance_pp. DOES halt
                                    the walk for frontier purposes.
  "supported"                    -- paired N adequate, delta_br within
                                    tolerance.
  "not_evaluated_beyond_breach"  -- every tier ordinally above the first
                                    confirmed risk_breach; raw stats still
                                    computed/reported for audit, excluded
                                    from both frontier summaries.
Confidence sub-label ("robust" if n_units_outcome_both >= ROBUST_MIN_N,
else "exploratory") applies only to "supported"/"risk_breach" rows.

TWO FRONTIER SUMMARIES PER BAND PER TOLERANCE, both ALWAYS initialized to
L0 (never null -- L0 was already an adequately-populated reference tier;
a band with no tested-and-passing candidate above it still has a valid
answer: "the reference tier, nothing more"):
  contiguous_supported_frontier_tier_ugx       -- the conservative,
      operationally-relevant number: advances from L0 only while every
      tier up to and including the candidate is "supported"; stops
      advancing (stays at the last accepted tier) at the first
      "risk_breach" OR "evidence_gap".
  highest_tier_with_any_supported_evidence_ugx -- advances to the highest
      tier EVER classified "supported", skipping over evidence gaps (but
      still capped below any risk_breach) -- surfaces non-contiguous
      higher-tier evidence separately rather than discarding it.
  first_risk_breach_tier_ugx                   -- null if none encountered.

TOLERANCE is a first-class SWEEP (--tolerances, default 2,3,4,5 pp), not a
single implied-correct default -- every output row carries its own
tolerance_pp. A band-level STABILITY summary reports whether
contiguous_supported_frontier_tier_ugx is the same at every swept tau; a
frontier that moves a lot across 2-5pp is reported as exactly that -- the
data not identifying a stable boundary -- never silently resolved by
picking one tolerance as truth.

ISOTONIC SMOOTHING is a secondary, NON-GATING diagnostic only (same
sklearn convention as scripts/fit_shadow_risk_calibration.py and scripts/
check_supported_exposure_boundary_robustness.py's smoothed method) --
reported alongside each band's result, never used to accept or reject a
tier.

NEVER in this script: assigned_limit as a feature or target anywhere;
PD/cal_pd; averaging float and commission into one score (kept fully
independent throughout, separate output files per fundamental).

Causal caveat (printed every run, same wording convention as every other
script in this workstream): this is observational, not causal evidence. A
tier classified "supported" at a given tolerance means historical
same-agent/same-anchor evidence did not show bad-rate deterioration beyond
that tolerance when observed at that tier -- it does NOT mean raising a
real agent's limit to that tier would be safe. Selection into exposure
tier, lender information, and policy changes all remain live explanations.

Restated independently (one-way scripts/ layering convention): the 7-tier
exposure set, _qcut_safe-style graceful degradation, the agent-period-unit
construction, and the same-agent/same-anchor pairing mechanic, consistent
with every other script in this rebuild.

Usage:
    python scripts\\derive_capacity_frontier_from_business_state.py ^
        --episode-dataset loan_episode_capacity_dataset.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

EXPOSURE_TIERS_UGX = [50_000, 100_000, 250_000, 350_000, 500_000, 750_000, 1_000_000]
FUNDAMENTALS_FOR_BANDS = {"float_activity_value_1m": "float", "commission": "commission"}
N_DECILES_DEFAULT = 10
EVIDENCE_GAP_MIN_N = 10   # paired N floor: below this, a candidate tier cannot be assessed at all.
ROBUST_MIN_N = 100        # paired N at/above which a classified tier's confidence is "robust".
BASELINE_MIN_N = 100      # marginal N required of a single tier before it may serve as L0.
DEFAULT_TOLERANCES_PP = [2.0, 3.0, 4.0, 5.0]

REQUIRED_COLS = [
    "agent_msisdn", "fundamentals_snapshot_date", "disbursement_amount_ugx",
    "float_activity_value_1m", "commission", "bad_state_3dpd_30d", "label_eligible_30d",
]


def _qcut_safe(s: pd.Series, q: int, prefix: str = "D") -> tuple:
    """qcut with duplicate bin edges dropped; falls back to as many bins as
    the data supports (minimum 1). Restated verbatim from
    analyze_business_state_exposure_variation.py (one-way scripts/
    layering convention). Returns (labeled band series, n_bins)."""
    try:
        codes, bins = pd.qcut(s, q, duplicates="drop", retbins=True, labels=False)
    except ValueError:
        codes, bins = None, None
    n_bins = (len(bins) - 1) if bins is not None else 0
    if n_bins <= 0:
        return pd.Series(f"{prefix}1", index=s.index), 1
    labels = [f"{prefix}{i + 1}" for i in range(n_bins)]
    return codes.map(dict(enumerate(labels))), n_bins


def build_agent_period_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (agent_msisdn, fundamentals_snapshot_date) unit.
    Restated subset of Level 2's version (median/p75/max_exposure dropped
    -- not needed here; EI is out of scope for a fundamentals-indexed
    frontier)."""
    key = ["agent_msisdn", "fundamentals_snapshot_date"]
    gb = df.groupby(key, sort=False)
    summary = pd.concat([
        gb.size().rename("n_loans_total"),
        gb["float_activity_value_1m"].first().rename("float_activity_value_1m"),
        gb["commission"].first().rename("commission"),
    ], axis=1).reset_index()

    elig_gb = df.loc[df["label_eligible_30d"] == 1].groupby(key, sort=False)["bad_state_3dpd_30d"]
    elig_summary = pd.concat(
        [elig_gb.size().rename("n_loans_eligible"), elig_gb.mean().rename("bad_rate_period")], axis=1
    ).reset_index()
    summary = summary.merge(elig_summary, on=key, how="left")
    summary["n_loans_eligible"] = summary["n_loans_eligible"].fillna(0).astype(int)

    known = df[df["disbursement_amount_ugx"].isin(EXPOSURE_TIERS_UGX)]
    tiers_summary = known.groupby(key, sort=False)["disbursement_amount_ugx"].nunique().rename(
        "distinct_tiers_experienced").reset_index()
    summary = summary.merge(tiers_summary, on=key, how="left")
    summary["distinct_tiers_experienced"] = summary["distinct_tiers_experienced"].fillna(0).astype(int)

    return summary[key + ["n_loans_total", "n_loans_eligible", "distinct_tiers_experienced",
                           "bad_rate_period", "float_activity_value_1m", "commission"]]


def assign_bands(agent_period_summary: pd.DataFrame, fund_col: str, n_bands: int = N_DECILES_DEFAULT) -> tuple:
    """Decile(-or-fewer)-of-unit's-own-fundamental assignment, computed
    once per unit. `n_bands` parameterized (not hardcoded) so a coarser
    override is possible if real-data deciles prove too thin."""
    valid_mask = agent_period_summary[fund_col].notna() & (agent_period_summary[fund_col] > 0)
    out = pd.Series(np.nan, index=agent_period_summary.index, dtype=object)
    if not valid_mask.any():
        return out, []
    band_series, n_bins = _qcut_safe(agent_period_summary.loc[valid_mask, fund_col], n_bands)
    labels = [f"D{i + 1}" for i in range(n_bins)]
    out.loc[valid_mask] = band_series.values
    return out, labels


def build_tier_marginal_table(df_with_bands: pd.DataFrame, band_col: str, band_labels: list) -> pd.DataFrame:
    """Loan-grain, restricted to the known 7-tier set. One row per
    (band, tier): n_units_at_tier (MARGINAL -- distinct agent-period units
    with >=1 loan at this tier in this band), n_loans, n_eligible, n_bad,
    bad_rate_loan_weighted."""
    cols = [band_col, "tier_ugx", "n_units_at_tier", "n_loans", "n_eligible", "n_bad", "bad_rate_loan_weighted"]
    known = df_with_bands[
        df_with_bands["disbursement_amount_ugx"].isin(EXPOSURE_TIERS_UGX) & df_with_bands[band_col].notna()
    ].copy()
    if known.empty:
        return pd.DataFrame(columns=cols)

    known["_bad_elig"] = known["bad_state_3dpd_30d"] * known["label_eligible_30d"]
    known["_unit_key"] = known["agent_msisdn"].astype(str) + "||" + known["fundamentals_snapshot_date"].astype(str)

    marginal = known.groupby([band_col, "disbursement_amount_ugx"], sort=False).agg(
        n_units_at_tier=("_unit_key", "nunique"),
        n_loans=("disbursement_amount_ugx", "size"),
        n_eligible=("label_eligible_30d", "sum"),
        n_bad=("_bad_elig", "sum"),
    ).reset_index().rename(columns={"disbursement_amount_ugx": "tier_ugx"})

    marginal["tier_ugx"] = marginal["tier_ugx"].astype(int)
    marginal["n_eligible"] = marginal["n_eligible"].astype(int)
    marginal["n_bad"] = marginal["n_bad"].astype(int)
    marginal["bad_rate_loan_weighted"] = np.where(
        marginal["n_eligible"] > 0, marginal["n_bad"] / marginal["n_eligible"], np.nan)
    return marginal[cols]


def select_baseline_tier(tier_marginal_band: pd.DataFrame, baseline_min_n: int):
    """L0 = the lowest EXPOSURE_TIERS_UGX rung with marginal
    n_units_at_tier >= baseline_min_n. Returns None if no tier in this
    band qualifies -- a single real tier, never a pooled blend."""
    lookup = dict(zip(tier_marginal_band["tier_ugx"], tier_marginal_band["n_units_at_tier"], strict=True))
    for tier in EXPOSURE_TIERS_UGX:
        if lookup.get(tier, 0) >= baseline_min_n:
            return tier
    return None


def build_unit_pair_rows(df_with_bands: pd.DataFrame) -> pd.DataFrame:
    """Restated verbatim from Level 2/Table C2's all-pairs mechanic
    (analyze_business_state_exposure_variation.py), with EI columns
    dropped (out of scope for a fundamentals-indexed frontier). One row
    per (agent_msisdn, fundamentals_snapshot_date, tier_low, tier_high)
    for every unordered pair of known tiers a unit's loans actually hit;
    units with <2 known tiers contribute zero rows."""
    key = ["agent_msisdn", "fundamentals_snapshot_date"]
    known = df_with_bands[df_with_bands["disbursement_amount_ugx"].isin(EXPOSURE_TIERS_UGX)].copy()
    known["_bad_elig"] = known["bad_state_3dpd_30d"] * known["label_eligible_30d"]

    per_tier = known.groupby(key + ["disbursement_amount_ugx"], sort=False).agg(
        n_loans=("disbursement_amount_ugx", "size"),
        n_eligible=("label_eligible_30d", "sum"),
        n_bad=("_bad_elig", "sum"),
    ).reset_index()
    per_tier["n_eligible"] = per_tier["n_eligible"].astype(int)
    per_tier["n_bad"] = per_tier["n_bad"].astype(int)
    per_tier["br"] = np.where(per_tier["n_eligible"] > 0, per_tier["n_bad"] / per_tier["n_eligible"], np.nan)

    multi_tier_units = per_tier.groupby(key, sort=False)["disbursement_amount_ugx"].transform("size") >= 2
    multi = per_tier[multi_tier_units]
    if multi.empty:
        return pd.DataFrame()

    merged = multi.merge(multi, on=key, suffixes=("_low", "_high"))
    merged = merged[merged["disbursement_amount_ugx_low"] < merged["disbursement_amount_ugx_high"]]

    out = merged.rename(columns={
        "disbursement_amount_ugx_low": "tier_low", "disbursement_amount_ugx_high": "tier_high",
    })[key + ["tier_low", "tier_high", "n_loans_low", "n_loans_high", "n_eligible_low", "n_eligible_high",
              "n_bad_low", "n_bad_high", "br_low", "br_high"]].copy()
    # disbursement_amount_ugx may load as float64 on a real CSV -- tier_low/tier_high are always
    # exact members of EXPOSURE_TIERS_UGX here, so this cast is lossless (same fix as the
    # confirmed real-run bug in analyze_business_state_exposure_variation.py's Table C2).
    out["tier_low"] = out["tier_low"].astype(int)
    out["tier_high"] = out["tier_high"].astype(int)
    out["delta_br"] = np.where(out["br_low"].notna() & out["br_high"].notna(), out["br_high"] - out["br_low"], np.nan)
    return out.reset_index(drop=True)


def attach_band_to_pairs(unit_pairs: pd.DataFrame, agent_period_summary: pd.DataFrame, band_col: str) -> pd.DataFrame:
    """Restated mechanic from Level 2's build_table_c2: merges each unit's
    OWN already-assigned band onto its pair rows -- never recomputed per
    tier, since the fundamentals snapshot (and therefore the band) is
    identical for every loan in a unit."""
    key = ["agent_msisdn", "fundamentals_snapshot_date"]
    if unit_pairs.empty:
        return unit_pairs.assign(**{band_col: pd.Series(dtype=object)})
    banded = unit_pairs.merge(agent_period_summary[key + [band_col]], on=key, how="left")
    return banded[banded[band_col].notna()]


def aggregate_baseline_pairs(pairs_for_band: pd.DataFrame, baseline_tier: int) -> pd.DataFrame:
    """Restated/renamed from Level 2's aggregate_unit_pairs / build_table_c2
    mechanic, filtered to tier_low == baseline_tier (the reference tier
    for this band): one row per candidate tier_high, with pooled
    (loan-weighted) and paired (within-unit) statistics kept distinct."""
    sub = pairs_for_band[pairs_for_band["tier_low"] == baseline_tier]
    if sub.empty:
        return pd.DataFrame()
    rows = []
    for th, g in sub.groupby("tier_high", sort=True):
        n_units_pair = len(g)
        n_loans_lower, n_loans_higher = int(g["n_loans_low"].sum()), int(g["n_loans_high"].sum())
        n_eligible_lower, n_eligible_higher = int(g["n_eligible_low"].sum()), int(g["n_eligible_high"].sum())
        n_bad_lower, n_bad_higher = int(g["n_bad_low"].sum()), int(g["n_bad_high"].sum())
        bad_rate_lower = n_bad_lower / n_eligible_lower if n_eligible_lower else np.nan
        bad_rate_higher = n_bad_higher / n_eligible_higher if n_eligible_higher else np.nan
        delta_br_loan_weighted_pp = (
            (bad_rate_higher - bad_rate_lower) * 100
            if pd.notna(bad_rate_lower) and pd.notna(bad_rate_higher) else np.nan
        )

        both = g[g["br_low"].notna() & g["br_high"].notna()]
        n_units_outcome_both = len(both)
        if n_units_outcome_both:
            deltas_pp = both["delta_br"] * 100
            median_delta_bad_rate_pp = deltas_pp.median()
            pct_delta_bad_positive = (deltas_pp > 0).mean() * 100
            pct_delta_bad_zero = (deltas_pp == 0).mean() * 100
            pct_delta_bad_negative = (deltas_pp < 0).mean() * 100
        else:
            median_delta_bad_rate_pp = pct_delta_bad_positive = pct_delta_bad_zero = pct_delta_bad_negative = np.nan

        rows.append({
            "candidate_tier_ugx": int(th), "n_units_pair": n_units_pair,
            "n_loans_lower": n_loans_lower, "n_loans_higher": n_loans_higher,
            "n_eligible_lower": n_eligible_lower, "n_eligible_higher": n_eligible_higher,
            "bad_rate_lower_loan_weighted": bad_rate_lower, "bad_rate_higher_loan_weighted": bad_rate_higher,
            "delta_br_loan_weighted_pp": delta_br_loan_weighted_pp,
            "n_units_outcome_both": n_units_outcome_both,
            "median_delta_bad_rate_pp": median_delta_bad_rate_pp,
            "pct_delta_bad_positive": pct_delta_bad_positive,
            "pct_delta_bad_zero": pct_delta_bad_zero,
            "pct_delta_bad_negative": pct_delta_bad_negative,
        })
    return pd.DataFrame(rows).sort_values("candidate_tier_ugx").reset_index(drop=True)


def classify_candidate_tiers(baseline_pairs_band: pd.DataFrame, baseline_tier: int,
                              tier_marginal_band: pd.DataFrame, tolerance_pp: float,
                              evidence_gap_min_n: int, robust_min_n: int) -> pd.DataFrame:
    """Walks EXPOSURE_TIERS_UGX ascending above baseline_tier, classifying
    each candidate independently. Returns the full per-tier audit table
    for one band at one tolerance, INCLUDING a leading "reference_tier"
    row for baseline_tier itself -- the printed/CSV sequence for a band
    must visibly start at L0, not at the first candidate."""
    baseline_marginal = tier_marginal_band[tier_marginal_band["tier_ugx"] == baseline_tier]
    baseline_marginal_br = float(baseline_marginal["bad_rate_loan_weighted"].iloc[0]) if not baseline_marginal.empty else np.nan

    rows = [{
        "candidate_tier_ugx": baseline_tier,
        "n_units_pair": np.nan, "n_units_outcome_both": np.nan,
        "bad_rate_lower_loan_weighted": baseline_marginal_br, "bad_rate_higher_loan_weighted": baseline_marginal_br,
        "delta_br_loan_weighted_pp": np.nan, "median_delta_bad_rate_pp": np.nan,
        "pct_delta_bad_positive": np.nan, "pct_delta_bad_zero": np.nan, "pct_delta_bad_negative": np.nan,
        "classification": "reference_tier", "confidence": "",
    }]

    pairs_lookup = ({int(r["candidate_tier_ugx"]): r for _, r in baseline_pairs_band.iterrows()}
                    if not baseline_pairs_band.empty else {})
    breach_hit = False
    for tier in EXPOSURE_TIERS_UGX:
        if tier <= baseline_tier:
            continue
        r = pairs_lookup.get(tier)
        if breach_hit:
            if r is None:
                continue  # never observed at all -- nothing to report, not even for audit
            rows.append({
                "candidate_tier_ugx": tier, "n_units_pair": r["n_units_pair"],
                "n_units_outcome_both": r["n_units_outcome_both"],
                "bad_rate_lower_loan_weighted": r["bad_rate_lower_loan_weighted"],
                "bad_rate_higher_loan_weighted": r["bad_rate_higher_loan_weighted"],
                "delta_br_loan_weighted_pp": r["delta_br_loan_weighted_pp"],
                "median_delta_bad_rate_pp": r["median_delta_bad_rate_pp"],
                "pct_delta_bad_positive": r["pct_delta_bad_positive"],
                "pct_delta_bad_zero": r["pct_delta_bad_zero"],
                "pct_delta_bad_negative": r["pct_delta_bad_negative"],
                "classification": "not_evaluated_beyond_breach", "confidence": "",
            })
            continue

        if r is None:
            # Never co-observed with the baseline under any anchor -- the most extreme
            # case of an evidence gap (n=0), not a distinct status.
            rows.append({
                "candidate_tier_ugx": tier, "n_units_pair": 0, "n_units_outcome_both": 0,
                "bad_rate_lower_loan_weighted": np.nan, "bad_rate_higher_loan_weighted": np.nan,
                "delta_br_loan_weighted_pp": np.nan, "median_delta_bad_rate_pp": np.nan,
                "pct_delta_bad_positive": np.nan, "pct_delta_bad_zero": np.nan, "pct_delta_bad_negative": np.nan,
                "classification": "evidence_gap", "confidence": "",
            })
            continue

        n_paired = r["n_units_outcome_both"]
        if pd.isna(n_paired) or n_paired < evidence_gap_min_n:
            classification, confidence = "evidence_gap", ""
        elif pd.notna(r["delta_br_loan_weighted_pp"]) and r["delta_br_loan_weighted_pp"] > tolerance_pp:
            classification = "risk_breach"
            confidence = "robust" if n_paired >= robust_min_n else "exploratory"
            breach_hit = True
        else:
            classification = "supported"
            confidence = "robust" if n_paired >= robust_min_n else "exploratory"

        rows.append({
            "candidate_tier_ugx": tier, "n_units_pair": r["n_units_pair"],
            "n_units_outcome_both": r["n_units_outcome_both"],
            "bad_rate_lower_loan_weighted": r["bad_rate_lower_loan_weighted"],
            "bad_rate_higher_loan_weighted": r["bad_rate_higher_loan_weighted"],
            "delta_br_loan_weighted_pp": r["delta_br_loan_weighted_pp"],
            "median_delta_bad_rate_pp": r["median_delta_bad_rate_pp"],
            "pct_delta_bad_positive": r["pct_delta_bad_positive"],
            "pct_delta_bad_zero": r["pct_delta_bad_zero"],
            "pct_delta_bad_negative": r["pct_delta_bad_negative"],
            "classification": classification, "confidence": confidence,
        })
    return pd.DataFrame(rows)


def summarize_band_frontier(classified: pd.DataFrame) -> dict:
    """Derives the two frontier summaries (both initialized to L0, never
    null) plus first_risk_breach_tier_ugx from the per-tier classification
    table for one band at one tolerance."""
    ref_row = classified[classified["classification"] == "reference_tier"].iloc[0]
    baseline_tier = int(ref_row["candidate_tier_ugx"])

    contiguous = baseline_tier
    highest_any_supported = baseline_tier
    first_breach = np.nan
    contiguous_still_open = True

    candidates = classified[classified["classification"] != "reference_tier"].sort_values("candidate_tier_ugx")
    for _, row in candidates.iterrows():
        cls = row["classification"]
        tier = int(row["candidate_tier_ugx"])
        if cls == "supported":
            if contiguous_still_open:
                contiguous = tier
            highest_any_supported = tier
        elif cls == "risk_breach":
            if pd.isna(first_breach):
                first_breach = tier
            contiguous_still_open = False
        elif cls == "evidence_gap":
            contiguous_still_open = False
        elif cls == "not_evaluated_beyond_breach":
            pass  # already excluded from both summaries by construction

    return {
        "contiguous_supported_frontier_tier_ugx": contiguous,
        "highest_tier_with_any_supported_evidence_ugx": highest_any_supported,
        "first_risk_breach_tier_ugx": first_breach,
    }


def fit_smoothed_marginal_curve(tier_marginal_band: pd.DataFrame):
    """Secondary, NON-GATING diagnostic only: isotonic fit of marginal
    bad_rate_loan_weighted vs. tier value within one band, weighted by
    n_eligible -- same sklearn convention as fit_shadow_risk_calibration.py
    and check_supported_exposure_boundary_robustness.py's smoothed method.
    Returns None if fewer than 2 distinct tiers have a defined bad rate."""
    valid = tier_marginal_band[tier_marginal_band["bad_rate_loan_weighted"].notna()]
    if valid["tier_ugx"].nunique() < 2:
        return None
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    iso.fit(valid["tier_ugx"].astype(float), valid["bad_rate_loan_weighted"], sample_weight=valid["n_eligible"])
    return iso


def smoothed_crosscheck_delta_pp(iso, baseline_tier: int, frontier_tier) -> float:
    """Descriptive-only: smoothed curve's predicted bad-rate delta (pp)
    between the frontier tier and the baseline tier. Never gating."""
    if iso is None or pd.isna(frontier_tier) or int(frontier_tier) == baseline_tier:
        return np.nan
    pred_baseline = iso.predict([float(baseline_tier)])[0]
    pred_frontier = iso.predict([float(frontier_tier)])[0]
    return (pred_frontier - pred_baseline) * 100


def derive_frontier_for_band(band: str, band_rank: int, tier_marginal_band: pd.DataFrame,
                              pairs_for_band: pd.DataFrame, tolerance_pp: float,
                              baseline_min_n: int, evidence_gap_min_n: int, robust_min_n: int) -> tuple:
    """Core per-band, per-tolerance orchestration. Returns (summary_row
    dict, classification_df for this band x tolerance)."""
    baseline_tier = select_baseline_tier(tier_marginal_band, baseline_min_n)
    if baseline_tier is None:
        return {
            "status": "insufficient_data_for_baseline_tier",
            "band": band, "band_rank": band_rank, "tolerance_pp": tolerance_pp,
            "baseline_tier_ugx": np.nan,
            "contiguous_supported_frontier_tier_ugx": np.nan,
            "highest_tier_with_any_supported_evidence_ugx": np.nan,
            "first_risk_breach_tier_ugx": np.nan,
            "smoothed_crosscheck_delta_at_frontier_pp": np.nan,
        }, pd.DataFrame()

    baseline_pairs = aggregate_baseline_pairs(pairs_for_band, baseline_tier)
    classified = classify_candidate_tiers(baseline_pairs, baseline_tier, tier_marginal_band,
                                           tolerance_pp, evidence_gap_min_n, robust_min_n)
    summary = summarize_band_frontier(classified)

    iso = fit_smoothed_marginal_curve(tier_marginal_band)
    crosscheck = smoothed_crosscheck_delta_pp(iso, baseline_tier, summary["contiguous_supported_frontier_tier_ugx"])

    baseline_n_units_marginal = int(
        tier_marginal_band.loc[tier_marginal_band["tier_ugx"] == baseline_tier, "n_units_at_tier"].iloc[0])

    classified = classified.copy()
    classified.insert(0, "band", band)
    classified.insert(1, "band_rank", band_rank)
    classified.insert(2, "baseline_tier_ugx", baseline_tier)
    classified.insert(3, "baseline_n_units_marginal", baseline_n_units_marginal)
    classified.insert(4, "baseline_min_n_used", baseline_min_n)
    classified.insert(5, "tolerance_pp", tolerance_pp)

    summary_row = {
        "status": "ok", "band": band, "band_rank": band_rank,
        "baseline_tier_ugx": baseline_tier, "tolerance_pp": tolerance_pp,
        "contiguous_supported_frontier_tier_ugx": summary["contiguous_supported_frontier_tier_ugx"],
        "highest_tier_with_any_supported_evidence_ugx": summary["highest_tier_with_any_supported_evidence_ugx"],
        "first_risk_breach_tier_ugx": summary["first_risk_breach_tier_ugx"],
        "smoothed_crosscheck_delta_at_frontier_pp": crosscheck,
    }
    return summary_row, classified


def compute_stability(summary_df: pd.DataFrame, band_labels: list, tolerances: list) -> pd.DataFrame:
    """Per band, across the swept tolerances: is
    contiguous_supported_frontier_tier_ugx stable? A band whose frontier
    moves a lot across tau is reported as exactly that, never silently
    resolved by picking one tolerance as truth."""
    rows = []
    for band in band_labels:
        band_rows = summary_df[(summary_df["band"] == band) & (summary_df["status"] == "ok")]
        base = {f"frontier_at_{tol}pp": np.nan for tol in tolerances}
        if band_rows.empty:
            rows.append({"band": band, "frontier_stable_across_tolerances": np.nan,
                         "frontier_tier_min_across_tolerances": np.nan,
                         "frontier_tier_max_across_tolerances": np.nan, **base})
            continue
        values = band_rows.set_index("tolerance_pp")["contiguous_supported_frontier_tier_ugx"]
        rows.append({
            "band": band,
            "frontier_stable_across_tolerances": bool(values.nunique() == 1),
            "frontier_tier_min_across_tolerances": values.min(),
            "frontier_tier_max_across_tolerances": values.max(),
            **{f"frontier_at_{tol}pp": values.get(tol, np.nan) for tol in tolerances},
        })
    return pd.DataFrame(rows)


def run_tolerance_sweep(df_with_bands: pd.DataFrame, agent_period_summary: pd.DataFrame,
                         band_col: str, band_labels: list, tolerances: list,
                         baseline_min_n: int, evidence_gap_min_n: int, robust_min_n: int) -> tuple:
    """PRIMARY entry point. Runs derive_frontier_for_band per band x every
    swept tolerance_pp, plus the cross-tolerance stability summary per
    band. Returns (summary_df, classification_df, stability_df,
    tier_marginals_df)."""
    tier_marginal_all = build_tier_marginal_table(df_with_bands, band_col, band_labels)
    unit_pairs = build_unit_pair_rows(df_with_bands)
    pairs_with_band = attach_band_to_pairs(unit_pairs, agent_period_summary, band_col)

    summary_rows = []
    classification_frames = []
    for band_rank, band in enumerate(band_labels, start=1):
        tier_marginal_band = tier_marginal_all[tier_marginal_all[band_col] == band]
        pairs_for_band = pairs_with_band[pairs_with_band[band_col] == band] if not pairs_with_band.empty else pairs_with_band
        for tol in tolerances:
            summary_row, classified = derive_frontier_for_band(
                band, band_rank, tier_marginal_band, pairs_for_band,
                tol, baseline_min_n, evidence_gap_min_n, robust_min_n,
            )
            summary_rows.append(summary_row)
            if not classified.empty:
                classification_frames.append(classified)

    summary_df = pd.DataFrame(summary_rows)
    classification_df = pd.concat(classification_frames, ignore_index=True) if classification_frames else pd.DataFrame()
    stability_df = compute_stability(summary_df, band_labels, tolerances)
    return summary_df, classification_df, stability_df, tier_marginal_all


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode-dataset", default="loan_episode_capacity_dataset.csv")
    ap.add_argument("--out-prefix", default="capacity_frontier")
    ap.add_argument("--tolerances", default=",".join(str(t) for t in DEFAULT_TOLERANCES_PP))
    ap.add_argument("--evidence-gap-min-n", type=int, default=EVIDENCE_GAP_MIN_N)
    ap.add_argument("--robust-min-n", type=int, default=ROBUST_MIN_N)
    ap.add_argument("--baseline-min-n", type=int, default=BASELINE_MIN_N)
    ap.add_argument("--n-bands", type=int, default=N_DECILES_DEFAULT)
    ap.add_argument("--print-detail", action="store_true",
                     help="Print the full per-tier classification audit table to stdout. Off by "
                          "default -- at real-data scale (many bands x candidate tiers x swept "
                          "tolerances) this table is large; it is always written to CSV regardless.")
    args = ap.parse_args(argv)

    path = Path(args.episode_dataset)
    if not path.exists():
        sys.exit(f"ERROR: {path} not found -- run build_loan_episode_capacity_dataset.py first.")
    df = pd.read_csv(path, low_memory=False)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: {path} is missing required column(s): {missing}")

    df_valid = df[df["fundamentals_snapshot_date"].notna()].copy()
    n_excluded = len(df) - len(df_valid)
    print(f"Loans with a valid measured business-state anchor: {len(df_valid):,} of {len(df):,} "
          f"({n_excluded:,} excluded -- no matched prior snapshot).")
    if df_valid.empty:
        sys.exit("ERROR: no loan has a valid measured business-state anchor -- nothing to analyze.")

    tolerances = sorted({float(x) for x in args.tolerances.split(",") if x.strip()})

    print(f"\n{'#' * 100}\nTERMINOLOGY (read before interpreting any number below)\n{'#' * 100}")
    print(
        "This script derives a HISTORICALLY-SUPPORTED EXPOSURE FRONTIER AT TOLERANCE TAU --\n"
        "estimated jointly from fundamentals (F), historical exposure tier (L), and subsequent\n"
        "outcome (Y). It is NOT Capacity(F). Capacity(F) is reserved for a later, separately-\n"
        "gated, FROZEN production mapping that -- once this frontier is derived and reviewed --\n"
        "takes only fundamentals as its runtime input."
    )

    agent_period_summary = build_agent_period_summary(df_valid)
    n_multi_tier = int((agent_period_summary["distinct_tiers_experienced"] >= 2).sum())
    print(f"\nBuilt {len(agent_period_summary):,} agent-period unit(s) across "
          f"{agent_period_summary['agent_msisdn'].nunique():,} agent(s). "
          f"{n_multi_tier:,} of them experienced >=2 distinct exposure tiers "
          f"(the population this frontier's paired evidence is drawn from).")

    for fund_col, fund_name in FUNDAMENTALS_FOR_BANDS.items():
        band_col = f"{fund_name}_band"
        agent_period_summary[band_col], band_labels = assign_bands(agent_period_summary, fund_col, args.n_bands)
        if not band_labels:
            print(f"\nWARNING: no valid {fund_name} bands (no unit has a positive, non-null "
                  f"{fund_col}) -- skipping {fund_name}.")
            continue

        df_with_bands = df_valid.merge(
            agent_period_summary[["agent_msisdn", "fundamentals_snapshot_date", band_col]],
            on=["agent_msisdn", "fundamentals_snapshot_date"], how="left",
        )

        print(f"\n{'#' * 100}\n# Historically-supported exposure frontier -- {fund_name}-banded\n{'#' * 100}")
        summary_df, classification_df, stability_df, tier_marginals_df = run_tolerance_sweep(
            df_with_bands, agent_period_summary, band_col, band_labels,
            tolerances, args.baseline_min_n, args.evidence_gap_min_n, args.robust_min_n,
        )

        classification_path = f"{args.out_prefix}_{fund_name}_frontier_classification.csv"
        summary_path = f"{args.out_prefix}_{fund_name}_frontier_summary.csv"
        stability_path = f"{args.out_prefix}_{fund_name}_frontier_stability.csv"
        tier_marginals_path = f"{args.out_prefix}_{fund_name}_tier_marginals.csv"
        classification_df.to_csv(classification_path, index=False)
        summary_df.to_csv(summary_path, index=False)
        stability_df.to_csv(stability_path, index=False)
        tier_marginals_df.to_csv(tier_marginals_path, index=False)

        print("-- Classification audit (read this BEFORE the summary table below) --")
        if classification_df.empty:
            print("  (no band produced a classification -- see status column in the summary table)")
        elif args.print_detail:
            with pd.option_context("display.max_rows", None, "display.width", 180, "display.float_format", "{:.3f}".format):
                print(classification_df.to_string(index=False))
        else:
            print(f"  {len(classification_df):,} row(s) written to {classification_path} -- not printed here "
                  f"(too large for a terminal at real-data scale: every band x candidate tier x swept "
                  f"tolerance). Open the CSV, or re-run with --print-detail to print it anyway "
                  f"(better piped to a file: ... > out.txt).")

        print("\n-- Frontier summary (per band x tolerance) --")
        with pd.option_context("display.max_rows", None, "display.width", 180, "display.float_format", "{:.3f}".format):
            print(summary_df.to_string(index=False))

        print("\n-- Stability across swept tolerances (per band) --")
        with pd.option_context("display.max_rows", None, "display.width", 180):
            print(stability_df.to_string(index=False))

    print(f"\n{'#' * 100}\nCausal caveat (printed every run)\n{'#' * 100}")
    print(
        "This is observational, not causal evidence. A tier classified 'supported' at a given\n"
        "tolerance means historical same-agent/same-anchor evidence did not show bad-rate\n"
        "deterioration beyond that tolerance when observed at that tier -- it does NOT mean\n"
        "raising a real agent's limit to that tier would be safe. Selection into exposure tier,\n"
        "lender information, and policy changes all remain live explanations throughout."
    )


if __name__ == "__main__":
    main()
