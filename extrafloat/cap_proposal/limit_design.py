"""
limit_design.py
===============
"Burden" limit design for the credit limit engine: the working baseline (run_engine_wrapper.py --limit-design burden, the default;
--limit-design current rolls back to the previous design).

What it changes, relative to the engine as it runs today (the engine source itself is NOT modified):
  1. The stale commission-tier ceiling is replaced, at every place the engine applies it (capacity step,
     thin-file cap, final policy step), by a per-borrower ceiling of  k x monthly commission
     (monthly commission = 6-month commission / 6).
  2. The experience ramp (risk cap x loans/10, floor 0.30) and the 100,000 thin-file cap are switched off and
     replaced by ONE explicit experience ladder: a cap of  k(stage) x monthly commission  whose multiple depends on
     completed loans, repayment record, severe-overdue history and current arrears.
  3. The engine's hard ceiling is raised (proposal: 5,000,000, the Bank of Uganda cap). Separately, the risk base limit (the scale of the
     risk cap: risk cap = risk score x risk base) is raised; it is a model setting, not a loan maximum, and its value is under review (see the
     risk base sweep in the calibration analysis).
  4. A demonstrated-exposure governor limits any increase over the 1,000,000-ceiling result to a multiple of the
     borrower's largest past loan, so higher limits are earned by history, not granted by commission alone.
  5. Audit columns (design_*) record the stage, the cap applied and the binding constraint for every limit.

Every threshold lives in limit_design_proposal.json. The values there are PROPOSALS for Credit Risk to set; the
reasoning for each is in the Credit Risk document. The thresholds are pending Credit Risk's confirmation; no customer-facing limit has been changed.
"""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULTS = {
    "capacity_ceiling_multiple": 4.0,
    "keep_peak_season_haircut": True,
    "engine_ceiling": 5_000_000,
    "risk_base_limit": 5_000_000,
    "governor": {"enabled": True, "multiple_of_largest_past_loan": 1.25, "reference_ceiling": 1_000_000, "reference_risk_base": 1_000_000},
    "ladder": {
        "no_completed_loan": 2.0,
        "early_clean": 3.0,
        "early_not_clean": 2.0,
        "established_streak": 4.0,
        "established_short_streak": 3.0,
        "established_severe_history": 2.0,
        "early_max_completed_loans": 2,
        "streak_required": 3,
        "severe_overdue_days": 30,
        # Which on-time rate decides "all completed loans on time". "prior" = the engine's on_time_repayment_rate (on-time rate over loans BEFORE the latest one;
        # undefined, and filled as 0, for a borrower with one loan). "lifetime" = lifetime_on_time_rate (every matured loan, including the latest).
        "clean_on_time_source": "prior",
    },
    "arrears": {
        "minor_max_days": 30, "minor_cap_multiple": 3.0, "severe_cap_multiple": 2.0, "block_lending_when_severe_overdue": False,
        # Where current arrears come from. "legacy" = the engine's active_loan_days_aging_at_snapshot (days past due as of the loan's last recorded event; the PD model's input).
        # "snapshot" = current_dpd_at_snapshot (31 Aug minus the due date of the latest unpaid loan) and written_off_at_snapshot, used by the credit-policy layer ONLY.
        # "any_loan" = worst_dpd_material_any_loan_at_snapshot: the worst days past due across EVERY loan that still carries a material balance (loans the business has
        # marked closed wrongly, W/C, and residuals under 5% of the loan or UGX 5,000 are ignored), plus written_off_at_snapshot. Credit policy only, like "snapshot".
        "source": "legacy",
        # With source "snapshot": severe current arrears = days past due AT OR ABOVE this value (>=), or a written-off latest loan; 1 to (value - 1) days is minor arrears.
        "severe_current_dpd_threshold": 30,
    },
    # Reactivation: a borrower returning after a long gap without any repayment does not immediately get the full experience multiple. The effective multiple is
    # min(experience multiple, cap_multiple) while days_since_last_repayment_any_loan_at_snapshot is above gap_days; the experience stage itself is not changed. It lifts when a
    # repayment is recorded (the gap resets). A borrower with no repayment date at all is NOT treated as dormant. Credit policy only: no PD change, no block.
    "reactivation": {"enabled": False, "gap_days": 150, "cap_multiple": 2.0},
    "risk_tier_multipliers_apply_to_first_exposure": True,
    "min_usable_loan": 300_000,
    # When a calibrated PD (cal_pd) is present: how the PD is turned into the engine's risk tiers, and which thin-file definition applies.
    "pd": {
        "tier_pd_cutoffs": [0.02, 0.05, 0.10],      # PD at or below 2% = tier 1, to 5% = tier 2, to 10% = tier 3, above 10% = tier 4. None = engine's 0.85/0.60/0.35 on (1 - PD)
        "thin_file_definition": "engine",           # "engine" = fewer than thin_file_loans_threshold loans; "pd" = the PD model's own flag (flags almost no borrower)
        "thin_file_loans_threshold": 3,
    },
}


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if str(k).startswith("_"):
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_design_config(path=None):
    """Defaults, overridden by the JSON settings file if it exists. Keys starting with '_' are comments."""
    d = copy.deepcopy(DEFAULTS)
    if path and Path(path).exists():
        d = _merge(d, json.loads(Path(path).read_text(encoding="utf-8-sig")))   # tolerate the BOM that Windows editors add
    return d


def stage_and_k(r, d):
    """Experience stage and applied multiple (x monthly commission) for every borrower in engine output/feature frame r."""
    L, A = d["ladder"], d["arrears"]
    n_early = int(L["early_max_completed_loans"])
    c = r["closed_loan_count"].fillna(0)
    on_time = r["lifetime_on_time_rate"] if L.get("clean_on_time_source", "prior") == "lifetime" and "lifetime_on_time_rate" in r.columns else r["on_time_repayment_rate"]
    clean = (r["lifetime_default_rate"].fillna(1) == 0) & (on_time.fillna(0) >= 1)   # "all completed loans on time"; unknown outcome counts as not on time
    sev = r["lifetime_worst_days_to_principal_cure"].fillna(0) >= L["severe_overdue_days"]
    streak = r["consecutive_on_time_loans"].fillna(0)
    early = c <= n_early
    names = {
        "S0": "S0 no completed loan",
        "S1": f"S1 1-{n_early} completed, all on time",
        "S1h": f"S1h 1-{n_early} completed, not all on time (hold)",
        "S1p": f"S1p 1-{n_early} completed, repayment outcome pending (hold)",
        "S2s": f"S2s {n_early + 1}+ completed, severe history",
        "S2a": f"S2a {n_early + 1}+ completed, on-time streak {int(L['streak_required'])}+",
        "S2b": f"S2b {n_early + 1}+ completed, streak under {int(L['streak_required'])}",
    }
    pending = early & r["lifetime_default_rate"].isna()       # no matured loan yet: not "failed", has not had the chance to show repayment (same multiple as the hold)
    stage = np.select([c == 0, early & clean, pending, early, (~early) & sev, (~early) & (streak >= L["streak_required"])],
                      [names["S0"], names["S1"], names["S1p"], names["S1h"], names["S2s"], names["S2a"]], names["S2b"])
    kmap = {names["S0"]: L["no_completed_loan"], names["S1"]: L["early_clean"], names["S1h"]: L["early_not_clean"], names["S1p"]: L["early_not_clean"],
            names["S2s"]: L["established_severe_history"], names["S2a"]: L["established_streak"], names["S2b"]: L["established_short_streak"]}
    stage = pd.Series(stage, index=r.index)
    k_stage = stage.map(kmap).astype(float)
    if A.get("source", "legacy") in ("snapshot", "any_loan"):
        age = r["worst_dpd_material_any_loan_at_snapshot" if A["source"] == "any_loan" else "current_dpd_at_snapshot"].fillna(0)
        anom = r["written_off_at_snapshot"].fillna(0) > 0
        thr = float(A.get("severe_current_dpd_threshold", 30))
        minor = (age > 0) & (age < thr)
        severe = (age >= thr) | anom
    else:
        age = r["active_loan_days_aging_at_snapshot"].fillna(0)
        anom = r["anomaly_open_at_snapshot"].fillna(0) > 0
        minor = (age > 0) & (age <= A["minor_max_days"])
        severe = (age > A["minor_max_days"]) | anom
    k = np.where(minor, np.minimum(k_stage, A["minor_cap_multiple"]), k_stage)
    k = np.where(severe, np.minimum(k, A["severe_cap_multiple"]), k)
    override = np.select([severe, minor & (k_stage > A["minor_cap_multiple"]), minor],
                         ["downgraded: severe overdue now", "held: minor overdue now", "overdue now (no change)"], "")
    return stage, pd.Series(k, index=r.index), pd.Series(override, index=r.index), names["S0"], severe


def _engine_pass(features, d, engine_ceiling, risk_base):
    """One engine run with: k x monthly-commission ceiling, thin-file cap and experience ramp off, given ceiling and risk base."""
    import extrafloat.engine.run_extrafloat_limit_engine as R
    from extrafloat.engine.extrafloat_limit_engine_caps import DEFAULT_CAP_CONFIG

    k = float(d["capacity_ceiling_multiple"])
    ceil = float(engine_ceiling)
    keep_season = bool(d["keep_peak_season_haircut"])
    cfg = copy.deepcopy(DEFAULT_CAP_CONFIG)
    cfg["combination"]["thin_file_max_tier"] = None           # thin-file hard cap replaced by the ladder
    cfg["experience"]["min_experience_factor"] = 1.0           # experience ramp replaced by the ladder
    cfg["global_ceiling_limit"] = ceil
    cfg["risk"]["base_limit"] = float(risk_base)
    f = features.copy()
    mc_ = f["commission"].fillna(0).clip(lower=0) / 6.0
    f["agent_tier_ceiling_multiplier"] = (k * mc_ / ceil).clip(0, 1)     # per-borrower ceiling at all three places the old tier ceiling applied
    orig = R.compute_capacity_cap

    def patched(df, c=None):
        out = orig(df, c)
        mc = out["commission"].fillna(0).clip(lower=0) / 6.0
        op = out["operational_activity_flag"].fillna(0).clip(0, 1)
        cr = out["recent_credit_active_flag"].fillna(0).clip(0, 1)
        act = 0.5 + 0.5 * (0.7 * op + 0.3 * cr)                # the engine's own activity multiplier is kept
        seas = out["capacity_season_multiplier"] if keep_season else 1.0
        out["capacity_cap"] = (k * mc * act * seas).clip(0, ceil)
        out["capacity_structural"] = (k * mc).clip(0, ceil)
        if not keep_season:
            out["capacity_season_multiplier"] = 1.0
        return out

    R.compute_capacity_cap = patched
    try:
        r = R.run_extrafloat_limit_engine(f, config=cfg)
    finally:
        R.compute_capacity_cap = orig
    return r


def _r100(x):
    return (np.asarray(x, dtype=float) / 100.0).round() * 100.0


import contextlib


@contextlib.contextmanager
def pd_tier_scale(d):
    """
    Re-express the policy risk tiers on a PD scale, for borrowers that have a calibrated PD (cal_pd).

    The engine sets tiers from risk_score with cut-offs 0.85 / 0.60 / 0.35, which were built for the 7-signal score. On the PD
    path the engine uses risk_score = 1 - PD, but the calibrated PD tops out near 16%, so the score never falls below ~0.84 and
    every borrower lands in tier 1 or 2. Here the tier score is re-mapped so that the SAME engine cut-offs fall at the chosen PD
    cut-offs (default 2% / 5% / 10%). Only the policy score changes: the risk CAP is still computed from 1 - PD, and the original
    value is kept in risk_score_pd. Borrowers without a PD keep the 7-signal score and its original cut-offs.
    """
    import extrafloat.engine.run_extrafloat_limit_engine as R

    cuts = (d.get("pd") or {}).get("tier_pd_cutoffs")
    if not cuts:
        yield
        return
    c1, c2, c3 = [float(x) for x in cuts]
    xs = [0.0, c1, c2, c3, max(0.5, 3 * c3)]
    ys = [1.0, 0.85, 0.60, 0.35, 0.0]
    orig = R.compute_risk_cap

    def patched(df, cfg=None):
        out = orig(df, cfg)
        if "cal_pd" in out.columns:
            has = out["cal_pd"].notna()
            out["risk_score_pd"] = out["risk_score"]
            out.loc[has, "risk_score"] = np.interp(out.loc[has, "cal_pd"].astype(float), xs, ys)
        return out

    R.compute_risk_cap = patched
    try:
        yield
    finally:
        R.compute_risk_cap = orig


def apply_thin_file_definition(features, d):
    """Use the engine's own thin-file definition (loans below a threshold) even when the PD model's flag was joined on."""
    p = d.get("pd") or {}
    if "cal_pd" in features.columns and p.get("thin_file_definition", "engine") == "engine" and "total_loans" in features.columns:
        f = features.copy()
        f["is_thin_file"] = (f["total_loans"].fillna(0) < p.get("thin_file_loans_threshold", 3)).astype(int)
        return f
    return features


def run_burden_design(features, d):
    """Run the burden design and return the engine result frame with assigned_limit replaced and design_* audit columns added."""
    ceil = float(d["engine_ceiling"])
    features = apply_thin_file_definition(features, d)
    with pd_tier_scale(d):
        r = _engine_pass(features, d, ceil, d["risk_base_limit"]).copy()
    if "risk_score_pd" in r.columns:                       # pd_tier_scale re-mapped risk_score for the tiers only: restore 1 - PD and keep the mapped score apart
        r["policy_risk_score"] = r["risk_score"]
        has_pd = r["risk_score_pd"].notna()
        r.loc[has_pd, "risk_score"] = r.loc[has_pd, "risk_score_pd"]
        r = r.drop(columns=["risk_score_pd"])
    mc = (r["commission"].fillna(0).clip(lower=0) / 6.0)
    stage, k, override, s0_name, severe = stage_and_k(r, d)
    k_before_react = k.copy()
    RA = d.get("reactivation") or {}
    gap_ = r["days_since_last_repayment_any_loan_at_snapshot"] if "days_since_last_repayment_any_loan_at_snapshot" in r.columns else pd.Series(np.nan, index=r.index)
    react = (gap_ > float(RA.get("gap_days", 150))) if RA.get("enabled") else pd.Series(False, index=r.index)       # NaN (never repaid) compares False
    if RA.get("enabled"):
        k = pd.Series(np.where(react, np.minimum(k, float(RA.get("cap_multiple", 2.0))), k), index=r.index)
    ladder_cap = k * mc
    engine_limit = r["assigned_limit"].astype(float).copy()
    if not d["risk_tier_multipliers_apply_to_first_exposure"]:
        ceil_k = np.minimum(d["capacity_ceiling_multiple"] * mc, ceil)
        engine_limit = pd.Series(np.where(stage == s0_name, r["combined_cap"].clip(upper=ceil_k), engine_limit), index=r.index)
    limit_ladder = pd.Series(_r100(np.minimum(engine_limit, ladder_cap)), index=r.index)

    G = d["governor"]
    if G["enabled"]:
        with pd_tier_scale(d):
            base = _engine_pass(features, d, G["reference_ceiling"], G["reference_risk_base"])
        base_limit = pd.Series(_r100(np.minimum(base["assigned_limit"].astype(float), ladder_cap)), index=r.index)
        hist = r["max_prior_loan_size"].fillna(0)
        gov_cap = pd.Series(np.maximum(base_limit, G["multiple_of_largest_past_loan"] * hist), index=r.index)
        final = pd.Series(_r100(np.minimum(limit_ladder, gov_cap)), index=r.index)
    else:
        base_limit = pd.Series(np.nan, index=r.index)
        gov_cap = pd.Series(np.nan, index=r.index)
        final = limit_ladder.copy()
    if RA.get("enabled"):
        lim_alt = pd.Series(_r100(np.minimum(engine_limit, k_before_react * mc)), index=r.index)
        final_alt = pd.Series(_r100(np.minimum(lim_alt, gov_cap)), index=r.index) if G["enabled"] else lim_alt
        react_reduces = (final_alt > final + 0.5)
    else:
        final_alt, react_reduces = final, pd.Series(False, index=r.index)
    before_block = final.copy()
    if d["arrears"]["block_lending_when_severe_overdue"]:
        final = final.where(~severe, 0.0)

    kmult = float(d["capacity_ceiling_multiple"])
    binding = np.select([final <= 0, final < limit_ladder - 1, (ladder_cap < engine_limit) & (k < kmult), ladder_cap < engine_limit],
                        ["zero (no commission, arrears block, or engine caps)", "demonstrated-exposure governor", "experience ladder", "commission ceiling"],
                        "engine caps (usage, exposure, risk, tier)")
    r["assigned_limit_before_design"] = r["assigned_limit"]
    r["assigned_limit"] = final
    r["design_stage"] = stage
    r["design_k_applied"] = k
    r["design_arrears_rule"] = override
    r["design_monthly_commission"] = mc
    r["design_commission_ceiling"] = np.minimum(kmult * mc, ceil)
    r["design_ladder_cap"] = ladder_cap
    r["design_limit_before_governor"] = limit_ladder
    r["design_governor_cap"] = gov_cap
    r["design_limit_at_reference_ceiling"] = base_limit
    r["design_binding_constraint"] = binding
    # decision waterfall: every constraint that reduced the limit, not only the last one
    src_ = d["arrears"].get("source", "legacy")
    snap_ = src_ in ("snapshot", "any_loan")
    dpd_ = (r["worst_dpd_material_any_loan_at_snapshot"] if src_ == "any_loan" else r["current_dpd_at_snapshot"]).fillna(0) if snap_ else r["active_loan_days_aging_at_snapshot"].fillna(0)
    r["design_current_dpd_at_snapshot"] = dpd_
    r["design_current_arrears_status"] = np.select([severe & (r["written_off_at_snapshot"].fillna(0) > 0 if snap_ else r["anomaly_open_at_snapshot"].fillna(0) > 0), severe, dpd_ > 0],
                                                   ["severe: written off", "severe: past due", "minor: past due"], "none")
    r["design_severe_current_arrears"] = severe.astype(int)
    r["design_arrears_block_applied"] = ((before_block > 0) & (final <= 0) & severe).astype(int)
    r["design_k_experience"] = k_before_react
    r["design_days_since_last_repayment_any_loan"] = gap_
    r["design_reactivation_cap_applied"] = react.astype(int)
    r["design_reactivation_binding"] = react_reduces.astype(int)
    r["design_limit_without_reactivation"] = final_alt
    r["design_ladder_binding"] = (ladder_cap < engine_limit - 1).astype(int)
    r["design_governor_binding"] = (final < limit_ladder - 1).astype(int)
    r["design_engine_risk_guardrail_binding"] = r["risk_cap_binding"].fillna(0).astype(int) if "risk_cap_binding" in r.columns else 0
    r["design_prior_limit_smoothing_applied"] = r["prior_limit_smoothing_applied"].fillna(0).astype(int) if "prior_limit_smoothing_applied" in r.columns else 0
    r["design_usable_loan"] = (final >= d["min_usable_loan"]).astype(int)
    # the calculated credit limit is kept as it is (assigned_limit); what can actually be drawn is a separate product question. A limit under the minimum drawable
    # loan is NOT lifted to the minimum: it stays a calculated limit with no drawable amount until a smaller product exists.
    r["design_calculated_credit_limit"] = final
    r["design_available_product_amount"] = final.where(final >= d["min_usable_loan"], 0.0)
    r["design_positive_below_minimum"] = ((final > 0) & (final < d["min_usable_loan"])).astype(int)
    return r
