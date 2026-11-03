import argparse
import multiprocessing
import sys

# freeze_support() required for a PyInstaller-frozen exe on Windows -- see
# run_segmentation_wrapper.py's comment for why.


def eligibility_from_loan_history(features, loan_history_path, rule, snapshot, use_snapshot_dpd=False, arrears_source="snapshot"):
    """Boolean Series (aligned to features) of borrowers allowed to benefit from the coverage fix.

    From loan_history_snapshot.csv at the same snapshot date:
      rule "no-overdue":   no loan with days-aging > 0 and no open rollover/anomaly loan.
      rule "no-open-loan": no unresolved (open) loan at all, and no open rollover/anomaly loan.
    With use_snapshot_dpd (design setting arrears.source = "snapshot") overdue means current_dpd_at_snapshot > 0 and a written-off latest loan blocks as well.
    A borrower missing from the snapshot cannot be verified and is NOT eligible.
    """
    import pandas as pd

    cols = ["msisdn", "snapshot_dt", "has_unresolved_loan_at_snapshot", "active_loan_days_aging_at_snapshot", "anomaly_open_at_snapshot"]
    if use_snapshot_dpd:
        cols += ["current_dpd_at_snapshot", "written_off_at_snapshot", "worst_dpd_material_any_loan_at_snapshot"] if arrears_source == "any_loan" else ["current_dpd_at_snapshot", "written_off_at_snapshot"]
    h = pd.read_csv(loan_history_path, usecols=cols, low_memory=False)
    dates = h["snapshot_dt"].astype(str).str.replace("-", "").str[:8].unique()
    if list(dates) != [snapshot]:
        sys.exit(f"ERROR: loan history snapshot_dt {list(dates)} does not match --snapshot {snapshot} ({loan_history_path}).")
    h["msisdn"] = h["msisdn"].astype(str)
    h = h.drop_duplicates("msisdn").set_index("msisdn")
    unresolved = h["has_unresolved_loan_at_snapshot"].fillna(0) > 0
    overdue = unresolved & (h["active_loan_days_aging_at_snapshot"].fillna(0) > 0)
    rollover = h["anomaly_open_at_snapshot"].fillna(0) > 0
    if use_snapshot_dpd:
        overdue = h["worst_dpd_material_any_loan_at_snapshot" if arrears_source == "any_loan" else "current_dpd_at_snapshot"].fillna(0) > 0
        rollover = rollover | (h["written_off_at_snapshot"].fillna(0) > 0)
    blocked = (unresolved if rule == "no-open-loan" else overdue) | rollover
    ok = ~blocked
    elig = features["msisdn"].astype(str).map(ok).fillna(False).astype(bool)
    print(f"coverage-fix eligibility ('{rule}'): {int(elig.sum()):,} eligible, {int((~elig).sum()):,} excluded "
          f"({int((~features['msisdn'].astype(str).isin(h.index)).sum())} not in loan history)")
    return elig


def apply_coverage_fix(features, mode, eligible=None):
    """Correction for the engine's repayment-coverage quirk (engine source is NOT modified).

    The engine computes coverage = repayment_val_1m / disbursement_val_1m. When the borrower disbursed
    nothing in the last month the division is undefined and the engine fills 0, which reads as "repaid
    nothing": recent_usage_coverage_multiplier drops to its 0.85 floor and prior_exposure's recent
    performance multiplier drops to its 0.70 floor.

    mode "narrow": only borrowers who REPAID last month with no new disbursement (repayment_val_1m > 0).
    mode "broad":  every borrower with no disbursement last month.
    In both, an undefined 1-month coverage is replaced by 3-month coverage (repayment_val_3m /
    disbursement_val_3m, capped at 1); if there was no disbursement in 3 months either, coverage is
    neutral (1.0). Borrowers who borrowed last month keep the engine's own value. Only raises limits.
    `eligible` (optional): only these borrowers are corrected; everyone else keeps the engine's own value
    (used to exclude borrowers with an overdue loan, so the fix never hides an unpaid obligation).
    """
    import numpy as np

    f = features.copy()
    d1, r1, d3, r3 = f["disbursement_val_1m"], f["repayment_val_1m"], f["disbursement_val_3m"], f["repayment_val_3m"]
    target = (d1 == 0) & ((r1 > 0) if mode == "narrow" else True)
    if eligible is not None:
        target = target & eligible.values
    fallback = np.where(d3 > 0, np.clip(r3 / d3.replace(0, np.nan), 0, 1), 1.0)
    cov = np.where(target, fallback, f["repayment_coverage_1m"].fillna(0))
    cnt = np.where(target, fallback, f["repayment_count_coverage_1m"].fillna(0))
    f["repayment_coverage_1m"] = cov
    f["recent_repayment_coverage_1m"] = cov
    f["repayment_count_coverage_1m"] = cnt
    strength = 0.7 * cov + 0.3 * cnt
    f["recent_repayment_strength"] = strength
    f["recent_repayment_performance"] = strength
    print(f"coverage fix '{mode}': {int(target.sum()):,} borrowers' coverage recomputed")
    return f


def install_capacity_scaling(mode):
    """Replace the engine's capacity scaling (engine source is NOT modified). Returns a restore function.

    The engine log-scales raw capacity against the MEDIAN tier ceiling of the batch being run
    (log1p(raw) / log1p(median ceiling) * median ceiling), then clips to each agent's own tier ceiling.
    That makes one agent's capacity depend on who else is in the batch (the median was 350,000 in one
    run and 1,000,000 in the next). Modes:
      "fixed-ref": same log scaling, but against the fixed global ceiling (1,000,000). Identical results
                   whenever the batch median ceiling is already 1,000,000; removes the batch dependence.
      "linear":    capacity = min(raw capacity, agent's tier ceiling). Policy change: lowers limits for
                   borrowers whose log-scaled capacity was generous relative to their raw capacity.
    The activity and peak-season multipliers are untouched: capacity_cap is rescaled by new/old structural.
    """
    import numpy as np

    import extrafloat.engine.run_extrafloat_limit_engine as R
    from extrafloat.engine.extrafloat_limit_engine_caps import DEFAULT_CAP_CONFIG

    ref = float(DEFAULT_CAP_CONFIG["global_ceiling_limit"])
    original = R.compute_capacity_cap

    def patched(df, cfg=None):
        out = original(df, cfg)
        raw = out["capacity_raw"].clip(lower=0)
        ceil = out["capacity_effective_ceiling"]
        if mode == "linear":
            new = np.minimum(raw, ceil)
        else:  # fixed-ref
            new = np.minimum(np.log1p(raw) / np.log1p(ref) * ref, ceil)
        old = out["capacity_structural"]
        out["capacity_cap"] = np.where(old > 0, out["capacity_cap"] * new / old.where(old > 0, 1.0), out["capacity_cap"])
        out["capacity_structural"] = new
        return out

    R.compute_capacity_cap = patched
    print(f"capacity scaling: '{mode}'")
    return lambda: setattr(R, "compute_capacity_cap", original)


if __name__ == "__main__":
    multiprocessing.freeze_support()

    import pandas as pd

    REPO = r"F:\AGENT DATA\AgentFloat-integrated-solution"
    OUT = r"F:\AGENT DATA\pipeline\output"
    sys.path.insert(0, REPO)

    ap = argparse.ArgumentParser(description="Run the credit limit engine for one agent-mart snapshot.")
    ap.add_argument("--snapshot", required=True, help="Agent mart date, YYYYMMDD. Must equal loan_summary_features snapshot_dt.")
    ap.add_argument("--transaction", help=r"Agent mart CSV. Default: F:\AGENT DATA\mfs_daily_agent_mart_<snapshot>.csv")
    ap.add_argument("--borrower", default=rf"{OUT}\borrower_history_features.csv")
    ap.add_argument("--loan-summary", default=rf"{OUT}\loan_summary_features.csv")
    ap.add_argument("--output", default=rf"{OUT}\engine_output.csv")
    ap.add_argument("--coverage-fix", choices=["off", "narrow", "broad"], default="narrow",
                    help="Zero-disbursement coverage quirk (see apply_coverage_fix). Default narrow; off = engine as shipped.")
    ap.add_argument("--coverage-eligibility", choices=["no-overdue", "no-open-loan", "none"], default="no-overdue",
                    help="Who may benefit from the coverage fix, judged from the loan history snapshot. Default no-overdue "
                         "(excludes borrowers with an overdue or rollover loan); no-open-loan is stricter; none = everyone.")
    ap.add_argument("--loan-history", default=rf"{OUT}\loan_history_snapshot.csv")
    ap.add_argument("--haircut-scope", choices=["open", "overdue"], default="overdue",
                    help="With the loan history on: 'overdue' (default) = only loans with days-aging > 0 are treated as unresolved; "
                         "'open' = engine's own design (any open loan, even a current one, takes the base haircut).")
    ap.add_argument("--use-loan-history", action=argparse.BooleanOptionalAction, default=True,
                    help="Feed the loan history snapshot to the engine so its risk cap applies the unresolved-loan haircut "
                         "(base 15%%, +1%% per day overdue up to 35%%, rollover 40%%). Default on; the engine otherwise "
                         "ignores overdue loans. --no-use-loan-history restores the previous behaviour.")
    ap.add_argument("--capacity-scaling", choices=["engine", "fixed-ref", "linear"], default="fixed-ref",
                    help="Capacity scaling (see install_capacity_scaling). Default fixed-ref; engine = as shipped; linear = policy change.")
    ap.add_argument("--limit-design", choices=["current", "burden"], default="burden",
                    help="'burden' (default, the working baseline) = ceiling of k x monthly commission, explicit experience ladder in place of "
                         "the ramp and thin-file cap, 5,000,000 engine ceiling, demonstrated-exposure governor, PD-scale risk tiers. "
                         "Thresholds come from --design-config (the defaults proposed to Credit Risk, pending their confirmation). "
                         "'current' = the design the engine ran before the baseline switch; use with --no-use-pd to reproduce it exactly.")
    ap.add_argument("--design-config", default=r"F:\AGENT DATA\pipeline\limit_design_proposal.json",
                    help="Settings for --limit-design burden (JSON). Edit this file to change thresholds; no code change is needed.")
    ap.add_argument("--use-pd", action=argparse.BooleanOptionalAction, default=True,
                    help="Score the PD model and use cal_pd in the engine's risk cap (default on). --no-use-pd = the engine's 7-signal risk score for everyone.")
    ap.add_argument("--pd-artifacts-dir", default=rf"{OUT}\pd_artifacts_labelC_train20260531_val20260831",
                    help="Trained PD model artifacts (checksum-verified before use). Default: the PD retrained on the confirmed outcome label C. "
                         "The frozen pre-correction model (original label) is output\\pd_artifacts_train20260531_val20260831: pass it explicitly to reproduce the frozen S0-S6 results.")
    ap.add_argument("--pd-champion", choices=["xgb", "lgb"], default="xgb")
    args = ap.parse_args()
    sys.path.insert(0, r"F:\AGENT DATA\pipeline")

    transaction = args.transaction or rf"F:\AGENT DATA\mfs_daily_agent_mart_{args.snapshot}.csv"

    # The engine joins borrowers to loan_summary on exact (msisdn, snapshot_dt). A
    # mismatched --snapshot does not fail: recent_usage_cap silently zero-fills for
    # 100% of borrowers (seen with the 09-22 mart). Fail loudly instead.
    ls_dates = pd.read_csv(args.loan_summary, usecols=["snapshot_dt"])["snapshot_dt"].astype(str).str.replace("-", "").str[:8].unique()
    if list(ls_dates) != [args.snapshot]:
        sys.exit(
            f"ERROR: --snapshot {args.snapshot} does not match loan_summary snapshot_dt {list(ls_dates)} in "
            f"{args.loan_summary}. Regenerate loan_summary_features.csv for that date first, or use the matching snapshot."
        )

    if args.coverage_fix == "off" and args.capacity_scaling == "engine" and not args.use_loan_history and args.limit_design == "current" and not args.use_pd:
        # Engine exactly as shipped.
        sys.argv = [
            "run_engine.py",
            "--transaction", transaction,
            "--borrower", args.borrower,
            "--loan-summary", args.loan_summary,
            "--output", args.output,
        ]
        from run_engine import main
        main()
    else:
        # Same steps as run_engine.main(), with the corrections applied around the engine.
        from pathlib import Path

        from extrafloat.engine.extrafloat_limit_engine_features import build_extrafloat_limit_engine_features
        from extrafloat.engine.run_extrafloat_limit_engine import run_extrafloat_limit_engine
        from extrafloat.io.extrafloat_data_loaders import (
            load_borrower_limit_features,
            load_loan_history_snapshot_features,
            load_transaction_capacity_features,
        )
        from run_engine import _load_loan_summary

        loan_history = None
        loan_history_pristine = None
        if args.use_loan_history or args.use_pd:
            loan_history = load_loan_history_snapshot_features(args.loan_history)
            # The engine joins this feed on msisdn only (no date check), so a stale file would be used silently.
            lh_dates = loan_history["snapshot_dt"].dt.strftime("%Y%m%d").dropna().unique()
            if list(lh_dates) != [args.snapshot]:
                sys.exit(f"ERROR: loan history snapshot_dt {list(lh_dates)} does not match --snapshot {args.snapshot} ({args.loan_history}).")
            loan_history_pristine = loan_history.copy()          # the PD model is scored on the loan history as received, as in the repo orchestrator
            if not args.use_loan_history:
                loan_history = None
        if args.use_loan_history:
            if args.haircut_scope == "overdue":
                od = (loan_history["has_unresolved_loan_at_snapshot"].fillna(0) > 0) & (
                    loan_history["active_loan_days_aging_at_snapshot"].fillna(0) > 0
                )
                print(f"haircut scope 'overdue': {int((loan_history['has_unresolved_loan_at_snapshot'] > 0).sum()):,} open loans -> {int(od.sum()):,} treated as unresolved")
                loan_history["has_unresolved_loan_at_snapshot"] = od.astype(int)
            print(f"loan history fed to engine: {len(loan_history):,} rows (unresolved-loan haircut ACTIVE)")
        features = build_extrafloat_limit_engine_features(
            load_borrower_limit_features(args.borrower),
            load_transaction_capacity_features(transaction),
            _load_loan_summary(args.loan_summary),
            loan_history,
        )
        if args.use_pd:
            import pd_join

            features, _ = pd_join.attach_pd(features, transaction, args.pd_artifacts_dir, loan_history_pristine, args.snapshot, champion=args.pd_champion)
        if args.coverage_fix != "off":
            elig = None
            if args.coverage_eligibility != "none":
                arrears_src = "legacy"
                if args.limit_design == "burden":
                    import limit_design as _ld

                    arrears_src = _ld.load_design_config(args.design_config)["arrears"].get("source", "legacy")
                elig = eligibility_from_loan_history(features, args.loan_history, args.coverage_eligibility, args.snapshot,
                                                     use_snapshot_dpd=arrears_src in ("snapshot", "any_loan"), arrears_source=arrears_src)
            features = apply_coverage_fix(features, args.coverage_fix, elig)
        restore = install_capacity_scaling(args.capacity_scaling) if args.capacity_scaling != "engine" else (lambda: None)
        try:
            if args.limit_design == "burden":
                import limit_design

                design = limit_design.load_design_config(args.design_config)
                print(f"limit design 'burden': ceiling {design['capacity_ceiling_multiple']}x monthly commission, engine ceiling {design['engine_ceiling']:,}, "
                      f"governor {'on (%.2fx largest past loan)' % design['governor']['multiple_of_largest_past_loan'] if design['governor']['enabled'] else 'off'}; settings from {args.design_config}")
                result = limit_design.run_burden_design(features, design)
                lim = result["assigned_limit"]
                print(f"burden design: total limit {lim.sum() / 1e6:,.0f}M | limits above 1,000,000: {(lim > 1e6).sum():,} | highest {lim.max():,.0f} | "
                      f"zero limits: {int((lim <= 0).sum()):,}")
            else:
                result = run_extrafloat_limit_engine(features)
        finally:
            restore()
        if "cal_pd" in result.columns:
            result["score_source"] = result["cal_pd"].notna().map({True: "PD model", False: "7-signal fallback"})
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)
        print(f"Output written to {args.output} ({len(result)} rows, {len(result.columns)} columns)")
