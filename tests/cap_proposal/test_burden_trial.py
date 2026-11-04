"""
Tests for the burden limit design trial (extrafloat/cap_proposal/).

The guarantees that matter most:
  1. running the design never changes what the live engine produces (same inputs -> same limits,
     before and after a design run), and the engine functions it swaps are always put back;
  2. the design's limits obey its own rules (ladder multiple x monthly commission, governor,
     engine ceiling, rounding).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

import extrafloat.engine.run_extrafloat_limit_engine as R
from extrafloat.cap_proposal import limit_design as ld
from extrafloat.cap_proposal import run_burden_trial as trial
from extrafloat.cap_proposal.synthetic_inputs import SNAPSHOT_DT, make_synthetic_inputs

logging.disable(logging.WARNING)


@pytest.fixture(scope="module")
def features(tmp_path_factory):
    raw = tmp_path_factory.mktemp("raw")
    paths = {}
    for name, frame in make_synthetic_inputs(400, seed=11).items():
        frame.to_csv(raw / f"{name}.csv", index=False)
        paths[name] = str(raw / f"{name}.csv")
    return trial.load_features(**paths, as_of_date=SNAPSHOT_DT)


@pytest.fixture(scope="module")
def settings():
    return ld.load_design_config(trial.DEFAULT_SETTINGS)


@pytest.fixture(scope="module")
def proposed(features, settings):
    return trial.run_proposed(features[1], settings)


# -- 1. live engine untouched ---------------------------------------------------


def test_live_engine_output_identical_before_and_after_design_run(features, settings):
    production, design = features
    before = trial.run_current(production)
    trial.run_proposed(design, settings)
    after = trial.run_current(production)
    pdt.assert_frame_equal(before, after)


def test_engine_functions_restored_after_design_run(features, settings):
    originals = (R.compute_capacity_cap, R.compute_risk_cap)
    trial.run_proposed(features[1], settings)
    assert (R.compute_capacity_cap, R.compute_risk_cap) == originals


def test_engine_functions_restored_when_design_raises(features, settings, monkeypatch):
    originals = (R.compute_capacity_cap, R.compute_risk_cap)

    def boom(*a, **k):
        raise RuntimeError("downstream failure")

    monkeypatch.setattr(ld, "stage_and_k", boom)
    with pytest.raises(RuntimeError, match="downstream failure"):
        trial.run_proposed(features[1], settings)
    assert (R.compute_capacity_cap, R.compute_risk_cap) == originals


def test_production_features_have_no_loan_history(features):
    production, design = features
    assert "closed_loan_count" not in production.columns
    assert "closed_loan_count" in design.columns


# -- 2. the design's own rules --------------------------------------------------


def test_limit_never_exceeds_ladder_cap_or_engine_ceiling(proposed, settings):
    lim = proposed["assigned_limit"]
    assert (lim <= proposed["design_ladder_cap"] + 50).all()        # +50: rounding to nearest 100
    assert (lim <= settings["engine_ceiling"]).all()
    assert (lim >= 0).all()
    assert (lim % 100 == 0).all()


def test_ladder_multiple_never_above_capacity_ceiling_multiple(proposed, settings):
    assert (proposed["design_k_applied"] <= settings["capacity_ceiling_multiple"]).all()


def test_governor_bounds_the_increase_over_the_reference_ceiling(proposed, settings):
    g = settings["governor"]["multiple_of_largest_past_loan"]
    gov = np.maximum(proposed["design_limit_at_reference_ceiling"], g * proposed["max_prior_loan_size"].fillna(0))
    assert (proposed["assigned_limit"] <= gov + 50).all()


def test_monthly_commission_is_six_month_commission_over_six(proposed):
    np.testing.assert_allclose(proposed["design_monthly_commission"], proposed["commission"].clip(lower=0) / 6.0)


def _stage_frame(**cols):
    base = {"closed_loan_count": 5, "lifetime_default_rate": 0.0, "on_time_repayment_rate": 1.0,
                "lifetime_worst_days_to_principal_cure": 0, "consecutive_on_time_loans": 5,
                "active_loan_days_aging_at_snapshot": 0, "anomaly_open_at_snapshot": 0}
    base.update(cols)
    return pd.DataFrame([base])


@pytest.mark.parametrize("cols, stage_prefix, k", [
    ({"closed_loan_count": 0}, "S0", 2.0),
    ({"closed_loan_count": 2}, "S1 ", 3.0),
    ({"closed_loan_count": 2, "on_time_repayment_rate": 0.5}, "S1h", 2.0),
    ({"closed_loan_count": 1, "lifetime_default_rate": np.nan}, "S1p", 2.0),
    ({"lifetime_worst_days_to_principal_cure": 45}, "S2s", 2.0),
    ({}, "S2a", 4.0),
    ({"consecutive_on_time_loans": 1}, "S2b", 3.0),
    ({"active_loan_days_aging_at_snapshot": 10}, "S2a", 3.0),        # minor arrears holds at 3x
    ({"active_loan_days_aging_at_snapshot": 45}, "S2a", 2.0),        # severe arrears drops to 2x
    ({"anomaly_open_at_snapshot": 1}, "S2a", 2.0),                   # open rollover counts as severe
])
def test_experience_ladder_stages(settings, cols, stage_prefix, k):
    stage, k_applied, *_ = ld.stage_and_k(_stage_frame(**cols), settings)
    assert stage.iloc[0].startswith(stage_prefix)
    assert k_applied.iloc[0] == k


# -- 3. trial plumbing ----------------------------------------------------------


def test_adapt_converts_worst_cure_hours_to_days(settings):
    f = _stage_frame().drop(columns="lifetime_worst_days_to_principal_cure")
    f["lifetime_worst_hours_to_principal_cure"] = 72.0
    f["commission"], f["max_prior_loan_size"] = 600_000.0, 100_000.0
    out = trial.adapt_for_design(f, settings)
    assert out["lifetime_worst_days_to_principal_cure"].iloc[0] == 3.0


def test_adapt_raises_on_missing_design_input(settings):
    f = _stage_frame(commission=600_000.0, max_prior_loan_size=0.0).drop(columns="closed_loan_count")
    with pytest.raises(ValueError, match="closed_loan_count"):
        trial.adapt_for_design(f, settings)


def test_compare_one_row_per_agent(features, settings, proposed):
    agents = trial.compare(trial.run_current(features[0]), proposed)
    assert agents["msisdn"].is_unique
    assert len(agents) == len(proposed)
    assert set(agents["change_direction"]) <= {"up", "down", "same", "not scored by current"}


def test_cli_synthetic_all_candidates(tmp_path):
    assert trial.main(["--synthetic", "300", "--all-candidates", "--out-dir", str(tmp_path)]) == 0
    idx = pd.read_csv(tmp_path / "index.csv")
    assert len(idx) == len(list(trial.HERE.glob("limit_design_*.json")))
    assert "skipped" not in idx.columns or idx["skipped"].isna().all()
    assert (tmp_path / "limit_design_proposal" / "summary.md").exists()


# -- 4. agent-category guard ----------------------------------------------------


def _with_guard(settings, mode):
    return {**settings, "agent_tier_guard": mode}


def test_default_guard_is_enforce(settings):
    assert settings["agent_tier_guard"] == "enforce"
    assert ld.DEFAULTS["agent_tier_guard"] == "enforce"


def test_enforce_keeps_every_limit_within_category_max(proposed):
    assert (proposed["assigned_limit"] <= proposed["design_agent_tier_max"] + 1).all()
    below = proposed["design_agent_category"] == "Below Threshold"
    assert below.any()
    assert (proposed.loc[below, "assigned_limit"] == 0).all()


def test_category_max_matches_engine_tier_table(proposed):
    from extrafloat.engine.extrafloat_limit_engine_caps import DEFAULT_CAP_CONFIG

    tiers = DEFAULT_CAP_CONFIG["agent_tier"]["tiers"]
    for cat, grp in proposed.groupby("design_agent_category"):
        expected = 0.0 if cat == "Below Threshold" else tiers[cat] * DEFAULT_CAP_CONFIG["global_ceiling_limit"]
        assert (grp["design_agent_tier_max"] == expected).all(), cat


def test_enforce_below_diamond_only_diamond_may_exceed(features, settings):
    r = trial.run_proposed(features[1], _with_guard(settings, "enforce_below_diamond"))
    over = r["assigned_limit"] > r["design_agent_tier_max"] + 1
    assert set(r.loc[over, "design_agent_category"]) <= {"diamond"}


def test_off_reproduces_design_without_guard(features, settings):
    r = trial.run_proposed(features[1], _with_guard(settings, "off"))
    pdt.assert_series_equal(r["assigned_limit"], r["design_limit_before_tier_guard"], check_names=False)
    assert r["design_agent_tier_guard_binding"].sum() == 0


def test_guard_never_raises_a_limit(features, settings, proposed):
    off = trial.run_proposed(features[1], _with_guard(settings, "off"))
    assert (proposed["assigned_limit"].values <= off["assigned_limit"].values + 1).all()


def test_invalid_guard_mode_raises(features, settings):
    with pytest.raises(ValueError, match="agent_tier_guard"):
        trial.run_proposed(features[1], _with_guard(settings, "sometimes"))


def test_cli_guard_modes_one_run_each(tmp_path):
    assert trial.main(["--synthetic", "300", "--agent-tier-guard", *trial.GUARD_MODES, "--out-dir", str(tmp_path)]) == 0
    idx = pd.read_csv(tmp_path / "index.csv").set_index("agent_tier_guard")
    assert set(idx.index) == set(trial.GUARD_MODES)
    assert idx.loc["enforce", "above_category_max"] == 0


def test_cli_guard_comparison_side_by_side(tmp_path):
    assert trial.main(["--synthetic", "300", "--agent-tier-guard", "enforce", "off", "enforce_below_diamond",
                       "--out-dir", str(tmp_path)]) == 0
    wide = pd.read_csv(tmp_path / "limit_design_proposal__guard_comparison_agents.csv")
    for col in ("limit_current", "limit_category_enforced", "limit_category_not_enforced", "limit_enforce_below_diamond"):
        assert col in wide.columns
    assert wide["msisdn"].is_unique
    assert (wide["limit_category_enforced"] <= wide["limit_category_not_enforced"] + 1).all()
    assert (wide["limit_category_enforced"] <= wide["category_max"] + 1).all()
    non_diamond = wide["agent_category"] != "diamond"
    pdt.assert_series_equal(wide.loc[non_diamond, "limit_enforce_below_diamond"],
                            wide.loc[non_diamond, "limit_category_enforced"], check_names=False)
    cat = pd.read_csv(tmp_path / "limit_design_proposal__guard_comparison_by_category.csv").set_index("agent_category")
    assert cat.loc["ALL AGENTS", "agents"] == len(wide)
    assert (tmp_path / "limit_design_proposal__guard_comparison.md").exists()


# -- 5. limits vs actual disbursements ------------------------------------------


def test_limits_vs_disbursed(tmp_path):
    from extrafloat.cap_proposal import compare_limits_vs_disbursed as cmp

    limits = pd.DataFrame({
        "msisdn": ["256700000001", "256700000002", "256700000003"],
        "agent_category": ["bronze", "diamond", "Below Threshold"],
        "category_max": [100_000, 1_000_000, 0],
        "limit_current": [100_000, 800_000, 0],
        "limit_category_enforced": [100_000, 1_000_000, 0],
        "limit_category_not_enforced": [150_000, 1_200_000, 20_000],
        "limit_enforce_below_diamond": [100_000, 1_200_000, 0],
    })
    disb = pd.DataFrame({
        "msisdn": ["256700000001", "256700000001", "256700000002", "256700000003", "256799999999"],
        "profile": "x", "month": ["2026-08", "2026-09", "2026-09", "2026-09", "2026-09"],
        "disbursed_amount": [50_000, 120_000, 900_000, 30_000, 10_000], "n_transactions": 1,
    })
    limits.to_csv(tmp_path / "agents.csv", index=False)
    disb.to_csv(tmp_path / "disb.csv", index=False)
    assert cmp.main(["--agents", str(tmp_path / "agents.csv"), "--disbursed", str(tmp_path / "disb.csv"),
                     "--out-dir", str(tmp_path / "out")]) == 0
    m = pd.read_csv(tmp_path / "out" / "limits_vs_disbursed_agents.csv", dtype={"msisdn": str}).set_index("msisdn")
    assert len(m) == 3                                                      # unmatched agent dropped
    assert m.loc["256700000001", "disbursed_amount"] == 120_000            # latest month used
    assert m.loc["256700000001", "limit_category_enforced__cut"] == 1      # took 120k, enforced limit 100k
    assert m.loc["256700000001", "limit_category_enforced__shortfall"] == 20_000
    assert m.loc["256700000001", "limit_category_not_enforced__cut"] == 0
    assert m.loc["256700000002", "limit_current__cut"] == 1
    assert m.loc["256700000003", "limit_category_not_enforced__cut"] == 1
    cat = pd.read_csv(tmp_path / "out" / "limits_vs_disbursed_by_category.csv")
    allrow = cat[(cat.agent_category == "ALL AGENTS") & (cat.limit == "category enforced")].iloc[0]
    assert allrow["cut_agents"] == 2 and allrow["agents_matched"] == 3

    assert cmp.main(["--agents", str(tmp_path / "agents.csv"), "--disbursed", str(tmp_path / "disb.csv"),
                     "--month", "2026-08", "--out-dir", str(tmp_path / "out8")]) == 0
    m8 = pd.read_csv(tmp_path / "out8" / "limits_vs_disbursed_agents.csv")
    assert len(m8) == 1


# -- 6. snapshot date alignment ---------------------------------------------------


def _write_synthetic(tmp_path, loan_summary_date=None):
    paths = {}
    for name, frame in make_synthetic_inputs(100, seed=3).items():
        if name == "loan_summary" and loan_summary_date:
            frame = frame.assign(snapshot_dt=loan_summary_date)
        frame.to_csv(tmp_path / f"{name}.csv", index=False)
        paths[name] = str(tmp_path / f"{name}.csv")
    return paths


def test_mismatched_loan_summary_date_is_refused(tmp_path):
    paths = _write_synthetic(tmp_path, loan_summary_date="20260817")
    with pytest.raises(trial.SnapshotDateMismatch, match="2026-08-17"):
        trial.load_features(**paths, as_of_date=SNAPSHOT_DT)
    args = ["--transaction", paths["transaction"], "--borrower", paths["borrower"], "--loan-summary", paths["loan_summary"],
            "--loan-history", paths["loan_history"], "--as-of-date", SNAPSHOT_DT, "--out-dir", str(tmp_path / "out")]
    assert trial.main(args) == 2
    assert trial.main([*args, "--allow-date-mismatch"]) == 0


def test_matching_dates_with_yyyymmdd_format_accepted(tmp_path):
    paths = _write_synthetic(tmp_path, loan_summary_date="20260831")   # same day, different text format
    production, _ = trial.load_features(**paths, as_of_date=SNAPSHOT_DT)
    assert len(production) > 0


def test_borrower_history_with_later_loans_is_refused(tmp_path):
    paths = _write_synthetic(tmp_path)
    bor = pd.read_csv(paths["borrower"])
    bor.loc[0, "latest_loan_ts"] = "2026-09-05"            # a loan after the 2026-08-31 snapshot
    bor.to_csv(paths["borrower"], index=False)
    with pytest.raises(trial.SnapshotDateMismatch, match="borrower history contains loans up to 2026-09-05"):
        trial.load_features(**paths, as_of_date=SNAPSHOT_DT)


def test_align_loan_dates_joins_later_loan_data(tmp_path):
    # mart 2026-07-31, loan summary + loan history 2026-08-17 (the real-data case)
    paths = {}
    for name, frame in make_synthetic_inputs(200, seed=5).items():
        if name == "transaction":
            frame = frame.assign(snapshot_dt="2026-07-31")
        elif name in ("loan_summary", "loan_history"):
            frame = frame.assign(snapshot_dt="20260817")
        elif name == "borrower":
            frame = frame.assign(latest_loan_ts="2026-08-15", latest_disbursement_ts="2026-08-15")
        frame.to_csv(tmp_path / f"{name}.csv", index=False)
        paths[name] = str(tmp_path / f"{name}.csv")
    with pytest.raises(trial.SnapshotDateMismatch):
        trial.load_features(**paths, as_of_date="2026-07-31")
    production, design = trial.load_features(**paths, as_of_date="2026-07-31", align_loan_dates_to_mart=True)
    assert (production["recent_disbursement_amount_1m"] > 0).any()        # loan summary actually joined
    assert "closed_loan_count" in design.columns
    assert any("17 days AFTER" in n for n in trial.DATA_NOTES)
    out = tmp_path / "out"
    args = ["--transaction", paths["transaction"], "--borrower", paths["borrower"], "--loan-summary", paths["loan_summary"],
            "--loan-history", paths["loan_history"], "--as-of-date", "2026-07-31", "--align-loan-dates", "--out-dir", str(out)]
    assert trial.main(args) == 0
    assert "Input note" in (out / "limit_design_proposal" / "summary.md").read_text()


def test_align_loan_dates_still_refuses_borrower_beyond_loan_date(tmp_path):
    paths = {}
    for name, frame in make_synthetic_inputs(100, seed=6).items():
        if name == "transaction":
            frame = frame.assign(snapshot_dt="2026-07-31")
        elif name in ("loan_summary", "loan_history"):
            frame = frame.assign(snapshot_dt="20260817")
        elif name == "borrower":
            frame = frame.assign(latest_loan_ts="2026-09-20")
        frame.to_csv(tmp_path / f"{name}.csv", index=False)
        paths[name] = str(tmp_path / f"{name}.csv")
    with pytest.raises(trial.SnapshotDateMismatch, match="borrower history"):
        trial.load_features(**paths, as_of_date="2026-07-31", align_loan_dates_to_mart=True)


# -- 7. frozen engine output (cal_pd + assigned_limit) ----------------------------


def test_frozen_engine_output_joins_pd_and_frozen_limit(tmp_path):
    paths = _write_synthetic(tmp_path)
    txn = pd.read_csv(paths["transaction"], dtype={"agent_msisdn": str})
    rng = np.random.default_rng(0)
    frozen = pd.DataFrame({"msisdn": txn["agent_msisdn"], "cal_pd": rng.uniform(0.005, 0.15, len(txn)),
                           "assigned_limit": 120_000.0, "risk_tier": "tier_1"})
    frozen.to_csv(tmp_path / "frozen.csv", index=False)
    base = ["--transaction", paths["transaction"], "--borrower", paths["borrower"], "--loan-summary", paths["loan_summary"],
            "--loan-history", paths["loan_history"], "--as-of-date", SNAPSHOT_DT,
            "--agent-tier-guard", "enforce", "off", "enforce_below_diamond"]
    assert trial.main([*base, "--out-dir", str(tmp_path / "no_pd")]) == 0
    assert trial.main([*base, "--frozen-engine-output", str(tmp_path / "frozen.csv"), "--out-dir", str(tmp_path / "pd")]) == 0

    a = pd.read_csv(tmp_path / "pd" / "limit_design_proposal__guard_enforce" / "agents.csv")
    assert a["cal_pd"].notna().all()
    assert (a["limit_frozen"] == 120_000).all()
    no_pd = pd.read_csv(tmp_path / "no_pd" / "limit_design_proposal__guard_enforce" / "agents.csv")
    assert "cal_pd" not in no_pd.columns or no_pd["cal_pd"].isna().all()
    assert not a["risk_tier"].equals(no_pd["risk_tier"])                   # PD changes the tiers
    wide = pd.read_csv(tmp_path / "pd" / "limit_design_proposal__guard_comparison_agents.csv")
    assert "limit_frozen" in wide.columns
    assert "frozen engine output" in (tmp_path / "pd" / "limit_design_proposal__guard_enforce" / "summary.md").read_text()

    from extrafloat.cap_proposal import compare_limits_vs_disbursed as cmp

    disb = wide[["msisdn"]].assign(profile="x", month="2026-08", disbursed_amount=100_000, n_transactions=1)
    disb.to_csv(tmp_path / "disb.csv", index=False)
    assert cmp.main(["--agents", str(tmp_path / "pd" / "limit_design_proposal__guard_comparison_agents.csv"),
                     "--disbursed", str(tmp_path / "disb.csv"), "--out-dir", str(tmp_path / "vs")]) == 0
    cat = pd.read_csv(tmp_path / "vs" / "limits_vs_disbursed_by_category.csv")
    assert "frozen engine" in set(cat["limit"])


# -- 8. tier preview (inspect_pd_source) -------------------------------------------


def test_tier_preview_shares_and_rules(tmp_path):
    from extrafloat.cap_proposal import inspect_pd_source as ips

    agents = pd.DataFrame({"msisdn": range(8), "cal_pd": [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.2, 0.5],
                           "thin_file_flag": [0, 0, 0, 0, 0, 0, 1, 1]})
    t = ips.tier_preview(agents, [0.02, 0.05, 0.10], None).set_index("tier")
    assert list(t["agents"]) == [2, 2, 2, 2]                     # boundaries inclusive, as in limit_design
    assert t.loc["tier_4", "thin_file_agents"] == 2
    assert abs(t["agent_share"].sum() - 1) < 1e-9
    assert ips.parse_cutoffs("3,12,30") == [0.03, 0.12, 0.30]
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        ips.parse_cutoffs("0.1,0.05,0.2")
    cuts = ips.cutoffs_for_shares(pd.Series(np.linspace(0, 1, 101)), ips.parse_shares("40,30,20,10"))
    np.testing.assert_allclose(cuts, [0.4, 0.7, 0.9])

    cal = pd.DataFrame({"model": "xgb", "score_min": [0, .5], "score_max": [.5, 1], "pd": [0.01, 0.30],
                        "n": [900, 100], "bad_rate": [0.01, 0.30]})
    t2 = ips.tier_preview(agents, [0.02, 0.05, 0.10], cal).set_index("tier")
    assert t2.loc["tier_1", "loan_share"] == 0.9 and t2.loc["tier_4", "loan_bad_rate"] == 0.30

    agents.to_csv(tmp_path / "out.csv", index=False)
    cal.to_csv(tmp_path / "cal.csv", index=False)
    assert ips.main(["--engine-output", str(tmp_path / "out.csv"), "--calibration-map", str(tmp_path / "cal.csv"),
                     "--cutoffs", "2,5,10", "--target-shares", "25,25,25,25", "--out", str(tmp_path / "p.csv")]) == 0
    assert len(pd.read_csv(tmp_path / "p.csv")) == 8


# -- 9. side-by-side summary of disbursement comparisons ---------------------------


def test_summarize_vs_disbursed(tmp_path):
    from extrafloat.cap_proposal import compare_limits_vs_disbursed as cmp
    from extrafloat.cap_proposal import summarize_vs_disbursed as summ

    disb = pd.DataFrame({"msisdn": ["1", "2"], "profile": "x", "month": "2026-08",
                         "disbursed_amount": [100_000, 300_000], "n_transactions": 1})
    disb.to_csv(tmp_path / "disb.csv", index=False)
    for name, lim in (("a", 200_000), ("b", 400_000)):
        pd.DataFrame({"msisdn": ["1", "2"], "agent_category": "gold", "category_max": 350_000,
                      "limit_frozen": 150_000, "limit_current": 250_000, "limit_category_enforced": lim,
                      "limit_category_not_enforced": lim, "limit_enforce_below_diamond": lim}
                     ).to_csv(tmp_path / f"{name}.csv", index=False)
        assert cmp.main(["--agents", str(tmp_path / f"{name}.csv"), "--disbursed", str(tmp_path / "disb.csv"),
                         "--out-dir", str(tmp_path / f"vs_{name}")]) == 0
    out = tmp_path / "summary"
    assert summ.main(["--run", f"A={tmp_path / 'vs_a'}", "--run", f"B={tmp_path / 'vs_b'}", "--out-dir", str(out)]) == 0
    ov = pd.read_csv(out / "summary_overall.csv", index_col=0)
    assert list(ov.columns) == ["frozen engine", "current engine", "A", "B"]
    assert ov.loc["cut_agents", "A"] == 1 and ov.loc["cut_agents", "B"] == 0       # 300k > 200k; nothing > 400k
    assert ov.loc["cut_agents", "frozen engine"] == 1
    assert (out / "summary.md").exists()
