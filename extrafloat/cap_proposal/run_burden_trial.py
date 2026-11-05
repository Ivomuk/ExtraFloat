"""
extrafloat/cap_proposal/run_burden_trial.py
===========================================
Side-by-side trial of the "burden" limit design (limit_design.py) against the
live engine, on the same agents. Read-only with respect to the live code:

  * the engine source is NOT modified -- limit_design.py swaps two engine
    functions in for the duration of its own call and restores them, and this
    runner checks they are restored after every design run;
  * the CURRENT limit is produced by run_extrafloat_limit_engine() exactly as
    run_engine.py runs it in production (no loan-history snapshot, no PD,
    no wrapper corrections) unless --current-with-loan-history is given;
  * nothing is written outside --out-dir.

Unlike run_engine_wrapper.py (the local Windows pipeline with hard-coded
F:\\ paths, PD join and coverage corrections), this runner needs only the
repo itself and the warehouse extracts, so it can be run anywhere.

Usage
-----
    # real extracts (loan history snapshot is required by the design's ladder)
    python -m extrafloat.cap_proposal.run_burden_trial \\
        --transaction  data/agent_profile_snapshot.csv \\
        --borrower     data/borrower_history.csv \\
        --loan-summary data/loan_summary.csv \\
        --loan-history data/loan_history_snapshot.csv \\
        --out-dir      output/burden_trial

    # every candidate settings file in extrafloat/cap_proposal/ in one go
    python -m extrafloat.cap_proposal.run_burden_trial ... --all-candidates

    # dry run on synthetic agents (checks the plumbing, not the numbers)
    python -m extrafloat.cap_proposal.run_burden_trial --synthetic 5000 --out-dir output/burden_trial_synth

Outputs (per settings file, in --out-dir/<settings name>/)
------------------------------------------------------------
    agents.csv     one row per agent: current vs proposed limit and the design audit columns
    summary.md     population totals, change distribution, stage and binding-constraint tables
    summary.json   the same figures, machine-readable
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import extrafloat.engine.run_extrafloat_limit_engine as R
from extrafloat.cap_proposal import limit_design as ld
from extrafloat.engine.extrafloat_limit_engine_features import build_extrafloat_limit_engine_features
from extrafloat.io.extrafloat_data_loaders import (
    load_borrower_limit_features,
    load_loan_history_snapshot_features,
    load_loan_summary_recent_features,
    load_transaction_capacity_features,
)

logger = logging.getLogger("burden_trial")

HERE = Path(__file__).resolve().parent
DEFAULT_SETTINGS = HERE / "limit_design_proposal.json"
LIVE_CEILING = 1_000_000.0

# Columns the design reads that the live engine does not need, by design setting.
_LADDER_COLS = ["commission", "closed_loan_count", "lifetime_default_rate", "on_time_repayment_rate",
                "lifetime_worst_days_to_principal_cure", "consecutive_on_time_loans", "max_prior_loan_size"]
_ARREARS_COLS = {
    "legacy": ["active_loan_days_aging_at_snapshot", "anomaly_open_at_snapshot"],
    "snapshot": ["current_dpd_at_snapshot", "written_off_at_snapshot"],
    "any_loan": ["worst_dpd_material_any_loan_at_snapshot", "written_off_at_snapshot"],
}

AGENT_COLUMNS = [
    "msisdn", "limit_frozen", "limit_current", "limit_proposed", "limit_change", "change_direction",
    "design_stage", "design_k_applied", "design_arrears_rule", "design_binding_constraint",
    "design_monthly_commission", "design_ladder_cap", "design_limit_before_governor",
    "design_governor_cap", "design_limit_at_reference_ceiling", "max_prior_loan_size",
    "is_thin_file", "risk_tier",
    "design_agent_category", "design_agent_tier_max", "design_agent_tier_guard",
    "design_limit_before_tier_guard", "design_agent_tier_guard_binding",
    # engine caps the design keeps: which of them sets the limit when design_binding_constraint = "engine caps"
    "capacity_cap", "recent_usage_cap", "prior_exposure_cap", "risk_cap",
    "combined_cap_before_risk_guardrail", "combined_cap", "combined_top_driver", "combined_reason",
    "policy_multiplier", "policy_reason", "risk_score", "cal_pd",
    # experience inputs and graduation (scenario settings with "graduation" enabled)
    "closed_loan_count", "consecutive_on_time_loans", "design_k_capacity", "design_graduation",
    "assigned_limit_before_design",
]
CATEGORY_ORDER = ["Below Threshold", "new bronze", "bronze", "silver", "gold", "platinum", "titanium", "diamond"]
GUARD_MODES = ("enforce", "enforce_below_diamond", "off", "enforce_except_graduates")


# -----------------------------------------------------------------------------
# Inputs
# -----------------------------------------------------------------------------


class SnapshotDateMismatch(ValueError):
    """The extracts are not all from the same snapshot date."""


def _dates(df: pd.DataFrame) -> list[str]:
    if df is None or "snapshot_dt" not in df.columns:
        return []
    return sorted(pd.to_datetime(df["snapshot_dt"], errors="coerce").dropna().dt.strftime("%Y-%m-%d").unique())


BORROWER_TS_COLUMNS = ("latest_loan_ts", "latest_disbursement_ts")


def borrower_latest_loan(borrower: pd.DataFrame) -> pd.Timestamp | None:
    """Latest loan timestamp in the borrower history -- its effective 'as of' date (the file has no snapshot_dt)."""
    ts = [pd.to_datetime(borrower[c], errors="coerce").max() for c in BORROWER_TS_COLUMNS if c in borrower.columns]
    ts = [t for t in ts if pd.notna(t)]
    return max(ts) if ts else None


# Caveats about the inputs of the current run, printed and written into every summary (reset per load).
DATA_NOTES: list[str] = []


def align_loan_dates(txn: pd.DataFrame, loan_summary: pd.DataFrame, loan_history: pd.DataFrame | None):
    """--align-loan-dates: relabel the loan summary / loan history snapshot_dt to the agent mart's date so the
    engine's exact-date join matches. The loan DATA is not changed -- it stays as of its own date. Returns
    (loan_summary, loan_history, latest original loan-data date or None)."""
    t = _dates(txn)
    if len(t) != 1:
        raise SnapshotDateMismatch(f"--align-loan-dates needs a single agent mart snapshot date; found {t}")
    target = pd.Timestamp(t[0])
    originals = []
    out = []
    for name, df in (("loan summary", loan_summary), ("loan history", loan_history)):
        if df is None:
            out.append(None)
            continue
        d = _dates(df)
        if len(d) != 1:
            raise SnapshotDateMismatch(f"--align-loan-dates needs a single {name} snapshot date; found {d}")
        if d[0] != t[0]:
            originals.append(pd.Timestamp(d[0]))
            gap = (pd.Timestamp(d[0]) - target).days
            when = "AFTER" if gap > 0 else "BEFORE"
            DATA_NOTES.append(f"{name} is as of {d[0]}, {abs(gap)} days {when} the agent mart ({t[0]}); its snapshot "
                              f"date was relabelled to {t[0]} so it joins (--align-loan-dates).")
            df = df.assign(snapshot_dt=target)
        out.append(df)
    if any(o > target for o in originals):
        DATA_NOTES.append("Loan features include activity after the agent mart date. Comparing these limits with "
                          f"disbursements made between {t[0]} and {max(originals):%Y-%m-%d} is biased towards agreement: "
                          "those loans are already in the inputs.")
    for n in DATA_NOTES:
        logger.warning("DATA NOTE: %s", n)
    return out[0], out[1], (max(originals) if originals else None)


def check_snapshot_dates(txn: pd.DataFrame, loan_summary: pd.DataFrame, loan_history: pd.DataFrame | None,
                         borrower: pd.DataFrame | None = None, borrower_max_date: pd.Timestamp | None = None) -> None:
    """Fail loudly when the extracts are from different snapshots.

    The engine joins the loan summary on (msisdn, snapshot_dt) with an EXACT date match and silently zero-fills
    every loan feature for unmatched agents (recent_usage_cap = 0, weaker risk score) -- a mismatched file does
    not error, it just produces wrong limits. The loan history is joined on msisdn only, so a stale file would be
    used without any warning at all."""
    t = _dates(txn)
    problems = []
    ls = _dates(loan_summary)
    if ls != t:
        problems.append(f"loan summary snapshot_dt {ls} != agent mart snapshot_dt {t}")
    lh = _dates(loan_history)
    if loan_history is not None and lh and lh != t:
        problems.append(f"loan history snapshot_dt {lh} != agent mart snapshot_dt {t}")
    if borrower is not None and len(t) == 1:
        latest = borrower_latest_loan(borrower)
        limit = max(pd.Timestamp(t[0]), borrower_max_date) if borrower_max_date is not None else pd.Timestamp(t[0])
        if latest is not None and latest.normalize() > limit:
            problems.append(f"borrower history contains loans up to {latest:%Y-%m-%d}, after the agent mart's "
                            f"snapshot {t[0]} (it was built for a later date; regenerate data/borrower_history.txt "
                            f"with snapshot_dt = {t[0].replace('-', '')})")
        elif latest is not None:
            logger.info("borrower history: latest loan %s (snapshot %s)", f"{latest:%Y-%m-%d}", t[0])
    if problems:
        raise SnapshotDateMismatch(
            "Input files are from different snapshot dates: " + "; ".join(problems) + ". "
            "Every loan feature would be zero-filled (recent usage cap = 0 for every agent). Regenerate the "
            "loan summary / loan history for the agent mart's date (data/loan_summary_query.txt, "
            "data/loan_history_snapshot_query.txt), or pass --align-loan-dates to join loan data from another "
            "date on purpose (recorded in the outputs)."
        )


def load_features(transaction: str, borrower: str, loan_summary: str, loan_history: str | None,
                  as_of_date: str | None = None, allow_date_mismatch: bool = False,
                  align_loan_dates_to_mart: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build engine features twice from the same extracts, with the live engine's own builder:
    (production_features, design_features). Production is what run_engine.py builds today
    (no loan history); the design needs the loan history snapshot for its ladder and arrears rules."""
    txn = load_transaction_capacity_features(transaction)
    bor = load_borrower_limit_features(borrower)
    ls = load_loan_summary_recent_features(loan_summary)
    lh = load_loan_history_snapshot_features(loan_history) if loan_history else None
    DATA_NOTES.clear()
    loan_date = None
    if align_loan_dates_to_mart:
        ls, lh, loan_date = align_loan_dates(txn, ls, lh)
    try:
        check_snapshot_dates(txn, ls, lh, bor, borrower_max_date=loan_date)
    except SnapshotDateMismatch as exc:
        if not allow_date_mismatch:
            raise
        logger.warning("RUNNING DESPITE DATE MISMATCH (--allow-date-mismatch): %s", exc)
        DATA_NOTES.append(f"Run with --allow-date-mismatch: {exc}")
    asof = pd.Timestamp(as_of_date) if as_of_date else None
    production = build_extrafloat_limit_engine_features(bor, txn, ls, None, as_of_date=asof)
    design = build_extrafloat_limit_engine_features(bor, txn, ls, lh, as_of_date=asof)
    return production, design


def adapt_for_design(features: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """Fill design inputs this repo's feature builder names differently, then check every input exists.

    The only adaptation: the repo's borrower history carries the worst cure time in HOURS
    (lifetime_worst_hours_to_principal_cure); the design reads DAYS. Converted as hours / 24.
    Anything else missing is an error -- the design must never silently fill an input."""
    f = features.copy()
    if "lifetime_worst_days_to_principal_cure" not in f.columns and "lifetime_worst_hours_to_principal_cure" in f.columns:
        f["lifetime_worst_days_to_principal_cure"] = f["lifetime_worst_hours_to_principal_cure"] / 24.0
        logger.info("design input: lifetime_worst_days_to_principal_cure = lifetime_worst_hours_to_principal_cure / 24")
    if (settings["ladder"].get("clean_on_time_source") == "lifetime" and "lifetime_on_time_rate" not in f.columns
            and "lifetime_on_time_24h_rate" in f.columns):
        # same 24h on-time definition the engine's on_time_repayment_rate uses (from prior_on_time_24h_rate),
        # over every matured loan instead of loans before the latest one
        f["lifetime_on_time_rate"] = f["lifetime_on_time_24h_rate"]
        logger.info("design input: lifetime_on_time_rate = lifetime_on_time_24h_rate")
    needed = list(_LADDER_COLS) + _ARREARS_COLS[settings["arrears"].get("source", "legacy")]
    if settings["ladder"].get("clean_on_time_source") == "lifetime":
        needed.append("lifetime_on_time_rate")
    if (settings.get("reactivation") or {}).get("enabled"):
        needed.append("days_since_last_repayment_any_loan_at_snapshot")
    missing = [c for c in needed if c not in f.columns]
    if missing:
        raise ValueError(
            f"burden design inputs missing from the features: {missing}. "
            "All of them come from the loan history snapshot (--loan-history); current_dpd_at_snapshot, "
            "written_off_at_snapshot, worst_dpd_material_any_loan_at_snapshot and "
            "days_since_last_repayment_any_loan_at_snapshot only from the newer extract, not from "
            "data/loan_history_snapshot_query.txt."
        )
    return f


# -----------------------------------------------------------------------------
# Runs
# -----------------------------------------------------------------------------


def run_current(features: pd.DataFrame) -> pd.DataFrame:
    """The live engine, unchanged."""
    return R.run_extrafloat_limit_engine(features)


ENGINE_OUTPUT_LEAD = ["msisdn", "assigned_limit", "risk_tier", "cal_pd", "score_source", "agent_category",
                      "design_agent_category", "design_binding_constraint", "design_stage", "design_graduation",
                      "design_k_applied", "design_k_capacity", "design_monthly_commission", "design_agent_tier_max",
                      "assigned_limit_before_design", "final_decision_reason"]


def engine_output_frame(proposed: pd.DataFrame) -> pd.DataFrame:
    """The design's full result in engine-output form: one row per agent, assigned_limit = the design's limit,
    every engine and design_* column kept (key columns first). assigned_limit_before_design = the engine's own
    limit inside the design run; final_decision_reason is the engine's reason, design_binding_constraint the design's."""
    out = proposed.copy()
    if "cal_pd" in out.columns and "score_source" not in out.columns:
        out["score_source"] = np.where(out["cal_pd"].notna(), "pd_model", "7_signal_fallback")
    lead = [c for c in ENGINE_OUTPUT_LEAD if c in out.columns]
    return out[lead + [c for c in out.columns if c not in lead]]


def write_engine_output(eo: Path, engine_runs: list, args, paths: dict, as_of) -> None:
    """Engine-form file for the FIRST category setting run (assigned_limit), with each other setting's limit added as
    limit_<setting> (e.g. limit_category_enforced, limit_category_not_enforced), plus FILE.provenance.json."""
    primary_mode, settings, proposed = engine_runs[0]
    frame = engine_output_frame(proposed)
    frame.insert(frame.columns.get_loc("assigned_limit") + 1, "assigned_limit_setting", primary_mode)
    key = frame["msisdn"].astype(str)
    totals = {primary_mode: round(float(frame["assigned_limit"].sum()) / 1e6, 1)}
    pos = frame.columns.get_loc("assigned_limit_setting") + 1
    for mode, _s, other in engine_runs[1:]:
        col = f"limit_{GUARD_LABELS[mode]}"
        frame.insert(pos, col, key.map(other.set_index(other["msisdn"].astype(str))["assigned_limit"]).values)
        pos += 1
        totals[mode] = round(float(frame[col].sum()) / 1e6, 1)
    eo.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(eo, index=False)
    prov = {
        "written_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "settings_file": str(next(iter(_settings_files(args)))),
        "assigned_limit_setting": primary_mode,
        "extra_limit_columns": {f"limit_{GUARD_LABELS[m]}": m for m, _s, _p in engine_runs[1:]},
        "total_limit_m_by_setting": totals,
        "tier_pd_cutoffs": (settings.get("pd") or {}).get("tier_pd_cutoffs"),
        "graduation_enabled": bool((settings.get("graduation") or {}).get("enabled")),
        "governor_multiple_of_largest_past_loan": settings["governor"]["multiple_of_largest_past_loan"],
        "engine_ceiling": settings["engine_ceiling"],
        "inputs": dict(paths),
        "as_of_date": as_of,
        "frozen_engine_output": args.frozen_engine_output,
        "input_notes": list(DATA_NOTES),
        "agents": int(len(frame)),
        "note": "Proposal output from the burden-design trial, not the live engine. assigned_limit = design limit "
                "under assigned_limit_setting; limit_* columns = the same design under the other category settings.",
    }
    Path(str(eo) + ".provenance.json").write_text(json.dumps(prov, indent=2, default=str))
    logger.info("engine output written: %s (%d rows, %d columns)", eo, len(frame), frame.shape[1])
    print(f"Engine output: {eo.resolve()}  (assigned_limit = {primary_mode}; totals by setting: {totals})")


def run_proposed(features: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """The burden design, with a check that the engine functions it swaps are put back."""
    originals = (R.compute_capacity_cap, R.compute_risk_cap)
    try:
        return ld.run_burden_design(adapt_for_design(features, settings), settings)
    finally:
        if (R.compute_capacity_cap, R.compute_risk_cap) != originals:
            R.compute_capacity_cap, R.compute_risk_cap = originals
            raise RuntimeError("limit_design left the live engine patched -- restored; results discarded.")


def _norm_msisdn(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.replace(r"[^0-9]", "", regex=True)


def load_frozen_engine_output(path: str) -> pd.DataFrame:
    """A previous engine run's output (e.g. output/engine_test_output.csv from run_credit_risk_pipeline.py):
    returns msisdn, cal_pd (if present) and limit_frozen (its assigned_limit), one row per agent."""
    f = pd.read_csv(path, dtype={"msisdn": str, "agent_msisdn": str}, low_memory=False)
    key = "msisdn" if "msisdn" in f.columns else "agent_msisdn" if "agent_msisdn" in f.columns else None
    if key is None:
        raise ValueError(f"{path}: no msisdn / agent_msisdn column")
    out = pd.DataFrame({"msisdn": _norm_msisdn(f[key])})
    if "cal_pd" in f.columns:
        out["cal_pd"] = pd.to_numeric(f["cal_pd"], errors="coerce")
    if "assigned_limit" in f.columns:
        out["limit_frozen"] = pd.to_numeric(f["assigned_limit"], errors="coerce")
    return out.drop_duplicates("msisdn", keep="last")


def attach_frozen_pd(features: pd.DataFrame, frozen: pd.DataFrame) -> pd.DataFrame:
    """Join the frozen run's cal_pd by msisdn, replacing any cal_pd already in the features. With cal_pd the engine
    uses 1 - PD for its risk cap (as run_credit_risk_pipeline.py does), and the design maps PD to its tier cut-offs."""
    if "cal_pd" not in frozen.columns:
        return features
    f = features.drop(columns=["cal_pd"], errors="ignore").copy()
    f["cal_pd"] = _norm_msisdn(f["msisdn"]).map(frozen.set_index("msisdn")["cal_pd"]).values
    return f


def compare(current: pd.DataFrame, proposed: pd.DataFrame, frozen: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per agent: current vs proposed limit plus the design's audit columns."""
    cur = current[["msisdn", "assigned_limit"]].rename(columns={"assigned_limit": "limit_current"})
    cur["msisdn"] = cur["msisdn"].astype(str)
    prop = proposed.rename(columns={"assigned_limit": "limit_proposed"}).copy()
    prop["msisdn"] = prop["msisdn"].astype(str)
    out = prop.merge(cur, on="msisdn", how="left", validate="one_to_one")
    if frozen is not None and "limit_frozen" in frozen.columns:
        out["limit_frozen"] = _norm_msisdn(out["msisdn"]).map(frozen.set_index("msisdn")["limit_frozen"]).values
    out["limit_change"] = out["limit_proposed"] - out["limit_current"]
    out["change_direction"] = np.select([out["limit_current"].isna(), out["limit_change"] > 0.5, out["limit_change"] < -0.5],
                                        ["not scored by current", "up", "down"], "same")
    return out[[c for c in AGENT_COLUMNS if c in out.columns]]


# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------


def category_table(agents: pd.DataFrame) -> pd.DataFrame:
    """Per agent category: the category maximum and how many proposed limits sit above it."""
    a = agents.assign(over=agents["limit_proposed"] > agents["design_agent_tier_max"] + 1,
                      excess=(agents["limit_proposed"] - agents["design_agent_tier_max"]).clip(lower=0))
    g = a.groupby("design_agent_category", dropna=False)
    t = pd.DataFrame({
        "agents": g.size(),
        "category_max": g["design_agent_tier_max"].max(),
        "max_current": g["limit_current"].max(),
        "max_proposed": g["limit_proposed"].max(),
        "total_current_m": g["limit_current"].sum() / 1e6,
        "total_proposed_m": g["limit_proposed"].sum() / 1e6,
        "above_category_max": g["over"].sum().astype(int),
        "excess_above_max_m": g["excess"].sum() / 1e6,
        "capped_by_guard": g["design_agent_tier_guard_binding"].sum().astype(int),
    })
    order = [c for c in CATEGORY_ORDER if c in t.index] + [c for c in t.index if c not in CATEGORY_ORDER]
    return t.reindex(order)


GUARD_LABELS = {
    "enforce": "category_enforced",
    "off": "category_not_enforced",
    "enforce_below_diamond": "enforce_below_diamond",
    "enforce_except_graduates": "enforce_except_graduates",
}


def guard_comparison(agents_by_mode: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Side by side for several agent_tier_guard modes of ONE settings file.
    Returns (per-agent wide table, per-category totals)."""
    modes = [m for m in ("enforce", "off", "enforce_below_diamond", "enforce_except_graduates") if m in agents_by_mode]
    first = agents_by_mode[modes[0]]
    wide = first[[c for c in ("msisdn", "design_agent_category", "design_agent_tier_max", "design_stage", "limit_frozen",
                              "limit_current") if c in first.columns]].copy()
    wide = wide.rename(columns={"design_agent_category": "agent_category", "design_agent_tier_max": "category_max",
                                "design_stage": "experience_stage"})
    for m in modes:
        lim = agents_by_mode[m].set_index("msisdn")["limit_proposed"]
        wide[f"limit_{GUARD_LABELS[m]}"] = wide["msisdn"].map(lim).values
    if "off" in modes:
        wide["above_category_max_if_not_enforced"] = (wide["limit_category_not_enforced"] > wide["category_max"] + 1).astype(int)

    order = [c for c in CATEGORY_ORDER if c in set(wide["agent_category"])]
    order += [c for c in wide["agent_category"].dropna().unique() if c not in order]
    g = wide.groupby("agent_category", dropna=False)
    cat = pd.DataFrame({"agents": g.size(), "category_max": g["category_max"].max()})
    if "limit_frozen" in wide.columns:
        cat["frozen_total_m"] = g["limit_frozen"].sum() / 1e6
    cat["current_total_m"] = g["limit_current"].sum() / 1e6
    for m in modes:
        col = f"limit_{GUARD_LABELS[m]}"
        cat[f"{GUARD_LABELS[m]}_total_m"] = g[col].sum() / 1e6
        cat[f"{GUARD_LABELS[m]}_max"] = g[col].max()
    if "off" in modes:
        cat["above_max_if_not_enforced"] = g["above_category_max_if_not_enforced"].sum()
    cat = cat.reindex(order)
    total = cat.sum(numeric_only=True)
    for c in cat.columns:
        if c.endswith("_max") or c == "category_max":
            total[c] = cat[c].max()
    cat.loc["ALL AGENTS"] = total
    return wide, cat


def render_guard_markdown(cat: pd.DataFrame, settings_name: str, synthetic: bool) -> str:
    note = ("\n> **Synthetic data.** Checks the mechanics only; says nothing about the real population.\n"
            if synthetic else "")
    note += "".join(f"\n> **Input note.** {n}\n" for n in DATA_NOTES)
    t = cat.reset_index().rename(columns={"index": "agent_category"})
    return (f"# Agent-category guard comparison: `{settings_name}`\n{note}\n"
            "Same agents, same design; only the agent-category guard differs. Totals in UGX millions.\n\n"
            + _md_table(t.round(1).to_dict("records")))


def _group_table(df: pd.DataFrame, by: str) -> pd.DataFrame:
    g = df.groupby(by, dropna=False)
    t = pd.DataFrame({
        "agents": g.size(),
        "total_current_m": g["limit_current"].sum() / 1e6,
        "total_proposed_m": g["limit_proposed"].sum() / 1e6,
        "median_current": g["limit_current"].median(),
        "median_proposed": g["limit_proposed"].median(),
        "share_up": g["change_direction"].apply(lambda s: (s == "up").mean()),
        "share_down": g["change_direction"].apply(lambda s: (s == "down").mean()),
    })
    t["share_of_agents"] = t["agents"] / len(df)
    return t.sort_values("agents", ascending=False)


def summarize(agents: pd.DataFrame, settings: dict, settings_name: str) -> dict:
    cur, prop = agents["limit_current"].fillna(0), agents["limit_proposed"]

    def pop(s: pd.Series) -> dict:
        return {
            "total_m": round(float(s.sum()) / 1e6, 1),
            "mean": round(float(s.mean()), 0),
            "median": round(float(s.median()), 0),
            "p90": round(float(s.quantile(0.9)), 0),
            "max": round(float(s.max()), 0),
            "zero": int((s <= 0).sum()),
            "above_1m": int((s > LIVE_CEILING).sum()),
        }

    total_cur = float(cur.sum())
    frozen = agents["limit_frozen"] if "limit_frozen" in agents.columns else None
    return {
        **({"frozen": pop(frozen.fillna(0)),
            "frozen_matched": int(frozen.notna().sum()),
            "total_change_vs_frozen_pct": round(100.0 * (float(prop.sum()) - float(frozen.fillna(0).sum()))
                                                / float(frozen.fillna(0).sum()), 1) if frozen.fillna(0).sum() > 0 else None}
           if frozen is not None else {}),
        "settings": settings_name,
        "agents": int(len(agents)),
        "current": pop(cur),
        "proposed": pop(prop),
        "total_change_pct": round(100.0 * (float(prop.sum()) - total_cur) / total_cur, 1) if total_cur > 0 else None,
        "direction": agents["change_direction"].value_counts().to_dict(),
        "by_stage": _group_table(agents, "design_stage").round(3).reset_index().to_dict("records"),
        "by_binding_constraint": _group_table(agents, "design_binding_constraint").round(3).reset_index().to_dict("records"),
        "by_agent_category": category_table(agents).round(3).reset_index().to_dict("records"),
        "above_category_max": int((agents["limit_proposed"] > agents["design_agent_tier_max"] + 1).sum()),
        "key_settings": {
            "capacity_ceiling_multiple": settings["capacity_ceiling_multiple"],
            "engine_ceiling": settings["engine_ceiling"],
            "governor": {k: v for k, v in settings["governor"].items() if not str(k).startswith("_")},
            "ladder": {k: v for k, v in settings["ladder"].items() if not str(k).startswith("_")},
            "arrears_source": settings["arrears"].get("source", "legacy"),
            "reactivation_enabled": bool((settings.get("reactivation") or {}).get("enabled")),
            "pd_tier_cutoffs": (settings.get("pd") or {}).get("tier_pd_cutoffs"),
            "agent_tier_guard": settings.get("agent_tier_guard", "enforce"),
        },
    }


def _md_table(records: list[dict]) -> str:
    if not records:
        return "_none_\n"
    cols = list(records[0].keys())
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in records:
        cells = []
        for c in cols:
            v = r[c]
            cells.append(f"{v:,.3f}" if isinstance(v, float) and abs(v) < 10 else f"{v:,.0f}" if isinstance(v, (int, float)) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def render_markdown(s: dict, synthetic: bool) -> str:
    rows = [{"metric": k, **({"frozen": s["frozen"][k]} if "frozen" in s else {}), "current": s["current"][k],
             "proposed": s["proposed"][k]} for k in s["current"]]
    note = ("\n> **Synthetic data.** These figures check that the design runs and that its rules fire as written. "
            "They say nothing about the real population.\n") if synthetic else ""
    note += "".join(f"\n> **Input note.** {n}\n" for n in DATA_NOTES)
    return (
        f"# Burden design trial: `{s['settings']}`\n{note}\n"
        f"Agents scored: **{s['agents']:,}**. Total limit change vs the live engine: **{s['total_change_pct']}%**"
        + (f"; vs the frozen engine output: **{s['total_change_vs_frozen_pct']}%** "
           f"({s['frozen_matched']:,} agents matched)" if "frozen" in s else "") + ".\n\n"
        f"## Population\n\n{_md_table(rows)}\n"
        f"Direction of change: {s['direction']}\n\n"
        f"## By agent category (guard: `{s['key_settings']['agent_tier_guard']}`)\n\n"
        f"Proposed limits above the agent's category maximum: **{s['above_category_max']:,}**.\n\n"
        f"{_md_table(s['by_agent_category'])}\n"
        f"## By experience stage\n\n{_md_table(s['by_stage'])}\n"
        f"## By binding constraint\n\n{_md_table(s['by_binding_constraint'])}\n"
        f"## Settings used\n\n```json\n{json.dumps(s['key_settings'], indent=2)}\n```\n"
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _settings_files(args) -> list[Path]:
    if args.all_candidates:
        return sorted(HERE.glob("limit_design_*.json"))
    return [Path(p) for p in (args.design_config or [DEFAULT_SETTINGS])]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Trial the burden limit design against the live engine (live code untouched).")
    src = ap.add_argument_group("inputs (warehouse extracts) -- or use --synthetic")
    src.add_argument("--transaction")
    src.add_argument("--borrower")
    src.add_argument("--loan-summary")
    src.add_argument("--loan-history", help="Loan history snapshot. Required by the design's experience ladder and arrears rules.")
    src.add_argument("--synthetic", type=int, metavar="N", help="Generate N synthetic agents instead of reading extracts.")
    ap.add_argument("--as-of-date", help="As-of date for feature engineering (YYYY-MM-DD); default today.")
    ap.add_argument("--design-config", nargs="+", help=f"Settings JSON(s). Default {DEFAULT_SETTINGS.name}.")
    ap.add_argument("--all-candidates", action="store_true", help="Run every limit_design_*.json in extrafloat/cap_proposal/.")
    ap.add_argument("--agent-tier-guard", nargs="+", choices=GUARD_MODES,
                    help="Override the settings' agent_tier_guard; several values run each settings file once per mode.")
    ap.add_argument("--frozen-engine-output",
                    help="A previous engine run's output CSV (e.g. output\\engine_test_output.csv from run.bat). Its cal_pd is "
                         "joined into BOTH runs (PD-based risk cap and tiers, as in run_credit_risk_pipeline.py) and its "
                         "assigned_limit is added as limit_frozen.")
    ap.add_argument("--align-loan-dates", action="store_true",
                    help="Join a loan summary / loan history from a DIFFERENT snapshot date on purpose: their snapshot_dt is "
                         "relabelled to the agent mart's date so the engine's exact-date join matches. The loan data itself "
                         "is unchanged. Recorded as an input note in every summary.")
    ap.add_argument("--allow-date-mismatch", action="store_true",
                    help="Run even if the loan summary / loan history snapshot_dt differs from the agent mart's. "
                         "Not recommended: unmatched agents get every loan feature zero-filled.")
    ap.add_argument("--current-with-loan-history", action="store_true",
                    help="Feed the loan history to the CURRENT run too (activates the engine's unresolved-loan haircut). "
                         "Default off = the engine exactly as run_engine.py runs it in production.")
    ap.add_argument("--out-dir", default="output/burden_trial")
    ap.add_argument("--engine-output", metavar="FILE",
                    help="Also write the design's full result in engine-output form (one row per agent, assigned_limit = "
                         "the new limit, all engine and design columns) to FILE, plus FILE.provenance.json. Needs exactly "
                         "one settings file. With several --agent-tier-guard values, assigned_limit uses the FIRST and the "
                         "others are added as limit_<setting> columns. Refuses to overwrite the --frozen-engine-output file.")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        from extrafloat.cap_proposal.synthetic_inputs import SNAPSHOT_DT, make_synthetic_inputs

        raw_dir = out_dir / "synthetic_inputs"
        raw_dir.mkdir(exist_ok=True)
        for name, frame in make_synthetic_inputs(args.synthetic).items():
            frame.to_csv(raw_dir / f"{name}.csv", index=False)
        paths = {k: str(raw_dir / f"{k}.csv") for k in ("transaction", "borrower", "loan_summary", "loan_history")}
        as_of = args.as_of_date or SNAPSHOT_DT
    else:
        need = {"--transaction": args.transaction, "--borrower": args.borrower, "--loan-summary": args.loan_summary,
                "--loan-history": args.loan_history}
        missing = [k for k, v in need.items() if not v]
        if missing:
            ap.error(f"missing {', '.join(missing)} (or use --synthetic N)")
        paths = {"transaction": args.transaction, "borrower": args.borrower,
                 "loan_summary": args.loan_summary, "loan_history": args.loan_history}
        as_of = args.as_of_date

    try:
        production, design_features = load_features(**paths, as_of_date=as_of, allow_date_mismatch=args.allow_date_mismatch,
                                                    align_loan_dates_to_mart=args.align_loan_dates)
    except SnapshotDateMismatch as exc:
        print(f"\nERROR: {exc}")
        return 2
    frozen = None
    if args.frozen_engine_output:
        frozen = load_frozen_engine_output(args.frozen_engine_output)
        production = attach_frozen_pd(production, frozen)
        design_features = attach_frozen_pd(design_features, frozen)
        if "cal_pd" in frozen.columns:
            cov = design_features["cal_pd"].notna().mean()
            DATA_NOTES.append(f"cal_pd joined from {args.frozen_engine_output} for {cov:.1%} of agents "
                              "(the rest use the engine's 7-signal risk score).")
        else:
            DATA_NOTES.append(f"{args.frozen_engine_output} has no cal_pd column: risk stays on the 7-signal score.")
        if "limit_frozen" in frozen.columns:
            matched = _norm_msisdn(design_features["msisdn"]).isin(frozen["msisdn"]).mean()
            DATA_NOTES.append(f"limit_frozen = assigned_limit from {args.frozen_engine_output} ({matched:.1%} of agents matched).")
        for n in DATA_NOTES[-2:]:
            logger.warning("DATA NOTE: %s", n)
    current = run_current(design_features if args.current_with_loan_history else production)

    index = []
    runs = []
    if args.engine_output:
        n_settings = len(_settings_files(args))
        if n_settings != 1:
            print(f"\nERROR: --engine-output needs exactly one settings file (got {n_settings}).")
            return 2
        if args.frozen_engine_output and Path(args.engine_output).resolve() == Path(args.frozen_engine_output).resolve():
            print("\nERROR: --engine-output would overwrite the --frozen-engine-output file (the baseline). Choose another name.")
            return 2
    for path in _settings_files(args):
        base = ld.load_design_config(path)
        if not path.exists():
            logger.warning("settings file %s not found -- using limit_design.py DEFAULTS", path)
        if args.agent_tier_guard:
            for mode in args.agent_tier_guard:
                runs.append((f"{path.stem}__guard_{mode}" if len(args.agent_tier_guard) > 1 else path.stem,
                             path.stem, {**base, "agent_tier_guard": mode}))
        else:
            runs.append((path.stem, path.stem, base))

    by_settings: dict[str, dict[str, pd.DataFrame]] = {}
    engine_runs: list[tuple[str, dict, pd.DataFrame]] = []
    for name, stem, settings in runs:
        try:
            proposed = run_proposed(design_features, settings)
        except ValueError as exc:
            logger.error("%s: SKIPPED -- %s", name, exc)
            index.append({"settings": name, "skipped": str(exc)})
            continue
        agents = compare(current, proposed, frozen)
        by_settings.setdefault(stem, {})[settings.get("agent_tier_guard", "enforce")] = agents
        summary = summarize(agents, settings, name)
        run_dir = out_dir / name
        run_dir.mkdir(exist_ok=True)
        agents.to_csv(run_dir / "agents.csv", index=False)
        if args.engine_output:
            engine_runs.append((settings.get("agent_tier_guard", "enforce"), settings, proposed))
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        (run_dir / "summary.md").write_text(render_markdown(summary, synthetic=bool(args.synthetic)))
        index.append({"settings": name, "agent_tier_guard": settings.get("agent_tier_guard", "enforce"), "agents": summary["agents"],
                      **({"total_frozen_m": summary["frozen"]["total_m"]} if "frozen" in summary else {}),
                      "total_current_m": summary["current"]["total_m"], "total_proposed_m": summary["proposed"]["total_m"],
                      "total_change_pct": summary["total_change_pct"], "above_1m": summary["proposed"]["above_1m"],
                      "zero_proposed": summary["proposed"]["zero"], "above_category_max": summary["above_category_max"]})
        logger.info("%s: total %.1fM -> %.1fM (%s%%), above 1M: %d", name, summary["current"]["total_m"],
                    summary["proposed"]["total_m"], summary["total_change_pct"], summary["proposed"]["above_1m"])

    if args.engine_output and engine_runs:
        write_engine_output(Path(args.engine_output), engine_runs, args, paths, as_of)

    for stem, agents_by_mode in by_settings.items():
        if len(agents_by_mode) < 2:
            continue
        wide, cat = guard_comparison(agents_by_mode)
        wide.to_csv(out_dir / f"{stem}__guard_comparison_agents.csv", index=False)
        cat.reset_index().rename(columns={"index": "agent_category"}).to_csv(
            out_dir / f"{stem}__guard_comparison_by_category.csv", index=False)
        (out_dir / f"{stem}__guard_comparison.md").write_text(render_guard_markdown(cat, stem, bool(args.synthetic)))
        print(f"\n== {stem}: agent-category guard comparison (UGX millions) ==")
        print(cat.round(1).to_string())

    idx = pd.DataFrame(index)
    idx.to_csv(out_dir / "index.csv", index=False)
    ran = idx[idx["skipped"].isna()].drop(columns="skipped") if "skipped" in idx else idx
    print(ran.to_string(index=False))
    if "skipped" in idx:
        for _, row in idx[idx["skipped"].notna()].iterrows():
            print(f"SKIPPED {row['settings']}: {row['skipped']}")
    for n in DATA_NOTES:
        print(f"INPUT NOTE: {n}")
    print(f"\nWritten to {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
