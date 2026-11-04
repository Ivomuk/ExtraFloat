"""
extrafloat/cap_proposal/inspect_pd_source.py
============================================
Read-only diagnostic: which model produced the cal_pd in an engine output, and why its scale looks the way it does.

    python -m extrafloat.cap_proposal.inspect_pd_source ^
        --engine-output output\\engine_test_output.csv ^
        --artifacts-dir pd_model\\artifacts

Reports
-------
1. The artifacts folder: which files exist, their sha256 against model_metadata.json, and the metadata itself
   (training date, git commit, label, versions) -- this identifies the model.
2. The calibration map (pd_calibration_map.csv): PD range per model. Thick-file cal_pd can never exceed its max.
3. Thin-file path: thin_file_lr.joblib present or not. Without it, thin-file cal_pd = never_loan_pd_like from the
   hand-set points scorecard + sigmoid (pd_model/modeling/scorecard.py), which is NOT calibrated to outcomes.
4. The engine output: cal_pd split by thin_file_flag / score_source, so the high PDs can be traced to a path.
Nothing is written or changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

EXPECTED = ["xgb_model.joblib", "lgbm_model.joblib", "feature_order.json", "pd_calibration_map.csv",
            "xgb_policy_thresholds.csv", "lgb_policy_thresholds.csv", "transform_report.csv", "model_metadata.json",
            "thin_file_lr.joblib", "thin_file_lr_features.json"]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def describe(s: pd.Series) -> str:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return "no values"
    q = s.quantile([0.25, 0.5, 0.75])
    return (f"n={len(s):,}  mean={s.mean():.4f}  min={s.min():.4f}  p25={q[0.25]:.4f}  median={q[0.5]:.4f}  "
            f"p75={q[0.75]:.4f}  max={s.max():.4f}  share>10%={(s > 0.10).mean():.1%}")


def inspect_artifacts(art: Path) -> dict:
    print(f"\n=== 1. Artifacts folder: {art.resolve()} ===")
    if not art.exists():
        print("  NOT FOUND")
        return {}
    meta = {}
    mpath = art / "model_metadata.json"
    if mpath.exists():
        try:
            meta = json.loads(mpath.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            print(f"  model_metadata.json is not valid JSON: {exc}")
    stored = meta.get("artifact_sha256", {}) if isinstance(meta, dict) else {}
    for f in EXPECTED:
        p = art / f
        if not p.exists():
            print(f"  {f:32s} MISSING")
            continue
        status = ""
        if f != "model_metadata.json":
            if f in stored:
                status = "sha256 OK" if stored[f] == _sha(p) else "sha256 DIFFERS from model_metadata.json"
            else:
                status = "no stored checksum"
        print(f"  {f:32s} present  {p.stat().st_size:>12,} bytes  modified {pd.Timestamp(p.stat().st_mtime, unit='s'):%Y-%m-%d %H:%M}  {status}")
    others = sorted(x.name for x in art.iterdir() if x.is_file() and x.name not in EXPECTED)
    if others:
        print(f"  other files: {', '.join(others[:20])}{' ...' if len(others) > 20 else ''}")
    if meta:
        print("\n  model_metadata.json (identifying fields):")
        for k, v in meta.items():
            if k in ("artifact_sha256",):
                continue
            text = json.dumps(v) if not isinstance(v, str) else v
            print(f"    {k}: {text[:300]}")
    return meta


def inspect_calibration(art: Path) -> None:
    print("\n=== 2. Calibration map (thick-file cal_pd comes from here) ===")
    p = art / "pd_calibration_map.csv"
    if not p.exists():
        print("  pd_calibration_map.csv MISSING")
        return
    cal = pd.read_csv(p)
    print(f"  columns: {list(cal.columns)}  rows: {len(cal)}")
    if {"model", "pd"} <= set(cal.columns):
        for m, g in cal.groupby("model"):
            pdv = pd.to_numeric(g["pd"], errors="coerce")
            print(f"  model={m}: {len(g)} bins, PD from {pdv.min():.4f} to {pdv.max():.4f}")
        print("  -> thick-file cal_pd can never be above that maximum.")


def inspect_thin_path(art: Path) -> None:
    print("\n=== 3. Thin-file path ===")
    if (art / "thin_file_lr.joblib").exists():
        print("  thin_file_lr.joblib present: thin-file cal_pd = isotonic-calibrated logistic regression, floored at 0.12.")
    else:
        print("  thin_file_lr.joblib MISSING: thin-file cal_pd = never_loan_pd_like from the hand-set points scorecard")
        print("  + sigmoid (pd_model/modeling/scorecard.py). It is NOT calibrated to outcomes and can be far above real bad rates.")


def inspect_output(path: Path) -> None:
    print(f"\n=== 4. Engine output: {path} ===")
    f = pd.read_csv(path, low_memory=False)
    print(f"  rows: {len(f):,}")
    pdcols = [c for c in f.columns if c.endswith("cal_pd") or c.endswith("raw_score") or c == "never_loan_pd_like"]
    print(f"  PD-related columns: {pdcols}")
    if "cal_pd" not in f.columns:
        print("  no cal_pd column")
        return
    print(f"  cal_pd overall:   {describe(f['cal_pd'])}")
    for col in ("thin_file_flag", "score_source"):
        if col in f.columns:
            for v, g in f.groupby(col, dropna=False):
                print(f"  cal_pd {col}={v}:  {describe(g['cal_pd'])}")
    if "thin_file_flag" in f.columns:
        thin = pd.to_numeric(f["thin_file_flag"], errors="coerce").eq(1)
        hi = pd.to_numeric(f["cal_pd"], errors="coerce") > 0.10
        if hi.any():
            print(f"  of agents with cal_pd > 10%: {(thin & hi).sum() / hi.sum():.1%} are thin-file")


# -----------------------------------------------------------------------------
# Tier preview: what a set of PD cut-offs does to the scored agents
# -----------------------------------------------------------------------------

TIER_MULTIPLIERS = {"tier_1": 1.00, "tier_2": 0.85, "tier_3": 0.65, "tier_4": 0.40}   # engine policy multipliers


def parse_cutoffs(text: str) -> list[float]:
    """'0.02,0.05,0.10' or '2,5,10' (percent) -> [0.02, 0.05, 0.10]; must be three increasing values in (0, 1)."""
    vals = [float(x) for x in text.replace(";", ",").split(",") if x.strip()]
    if len(vals) != 3:
        raise argparse.ArgumentTypeError(f"need three cut-offs (tier 1/2, 2/3, 3/4), got {text!r}")
    if any(v > 1 for v in vals):
        vals = [v / 100 for v in vals]
    if not (0 < vals[0] < vals[1] < vals[2] < 1):
        raise argparse.ArgumentTypeError(f"cut-offs must increase and lie between 0 and 1: {text!r}")
    return vals


def assign_tier(pd_values: pd.Series, cuts: list[float]) -> pd.Series:
    """Same rule as limit_design.pd_tier_scale: PD <= c1 tier 1, <= c2 tier 2, <= c3 tier 3, above tier 4."""
    c1, c2, c3 = cuts
    v = pd.to_numeric(pd_values, errors="coerce")
    t = pd.Series(pd.NA, index=v.index, dtype="object")
    t[v.notna()] = "tier_4"
    t[v <= c3] = "tier_3"
    t[v <= c2] = "tier_2"
    t[v <= c1] = "tier_1"
    return t


def tier_preview(agents: pd.DataFrame, cuts: list[float], cal_map: pd.DataFrame | None, model: str = "xgb") -> pd.DataFrame:
    """One row per tier: agents (share, expected bad rate = mean cal_pd), split thick/thin, and -- from the
    calibration map -- the share of the model's validation loans in that PD range and their observed bad rate."""
    pdv = pd.to_numeric(agents["cal_pd"], errors="coerce")
    tiers = assign_tier(pdv, cuts)
    thin = pd.to_numeric(agents["thin_file_flag"], errors="coerce").eq(1) if "thin_file_flag" in agents.columns else None
    edges = [0.0, *cuts, 1.0]
    rows = []
    n_pd = int(pdv.notna().sum())
    for i, t in enumerate(["tier_1", "tier_2", "tier_3", "tier_4"]):
        m = tiers.eq(t)
        row = {
            "tier": t,
            "pd_range": f"{edges[i]:.1%} - {edges[i + 1]:.1%}" if i < 3 else f"> {edges[i]:.1%}",
            "multiplier": TIER_MULTIPLIERS[t],
            "agents": int(m.sum()),
            "agent_share": m.sum() / n_pd if n_pd else float("nan"),
            "agent_expected_bad_rate": pdv[m].mean(),
        }
        if thin is not None:
            row["thin_file_agents"] = int((m & thin).sum())
        if cal_map is not None and {"model", "pd", "n", "bad_rate"} <= set(cal_map.columns):
            g = cal_map[cal_map["model"].astype(str) == model]
            lo, hi = edges[i], edges[i + 1]
            gm = g[(g["pd"] > lo) & (g["pd"] <= hi)] if i else g[g["pd"] <= hi]
            tot, bad_tot = g["n"].sum(), (g["n"] * g["bad_rate"]).sum()
            row["loan_share"] = gm["n"].sum() / tot if tot else float("nan")
            row["loan_bad_rate"] = (gm["n"] * gm["bad_rate"]).sum() / gm["n"].sum() if gm["n"].sum() else float("nan")
            row["share_of_all_bad_loans"] = (gm["n"] * gm["bad_rate"]).sum() / bad_tot if bad_tot else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def by_category(agents: pd.DataFrame, cuts: list[float]) -> pd.DataFrame | None:
    col = next((c for c in ("agent_category", "design_agent_category") if c in agents.columns), None)
    if col is None:
        return None
    t = assign_tier(agents["cal_pd"], cuts)
    return (pd.crosstab(agents[col], t, normalize="index", rownames=["agent_category"], colnames=["tier"]) * 100).round(1)


def _fmt(df: pd.DataFrame) -> str:
    out = df.copy()
    for c in out.columns:
        if c in ("agent_share", "agent_expected_bad_rate", "loan_share", "loan_bad_rate", "share_of_all_bad_loans"):
            out[c] = out[c].map(lambda v: "" if pd.isna(v) else f"{v:.1%}")
        elif c in ("agents", "thin_file_agents"):
            out[c] = out[c].map(lambda v: f"{v:,}")
    return out.to_string(index=False)


def parse_shares(text: str) -> list[float]:
    """'40,30,20,10' -> agent shares for tiers 1-4 (percent or fractions); must sum to 100%."""
    vals = [float(x) for x in text.replace(";", ",").split(",") if x.strip()]
    if len(vals) != 4:
        raise argparse.ArgumentTypeError(f"need four shares (tiers 1-4), got {text!r}")
    total = sum(vals)
    if abs(total - 100) < 1e-6:
        vals = [v / 100 for v in vals]
    elif abs(total - 1) > 1e-6:
        raise argparse.ArgumentTypeError(f"shares must sum to 100 (or 1): {text!r}")
    return vals


def cutoffs_for_shares(pd_values: pd.Series, shares: list[float]) -> list[float]:
    """PD cut-offs that put the requested share of agents in tiers 1-4 (agent PD quantiles)."""
    v = pd.to_numeric(pd_values, errors="coerce").dropna()
    cum = [sum(shares[: i + 1]) for i in range(3)]
    cuts = [float(v.quantile(c)) for c in cum]
    for i in (1, 2):                                   # keep strictly increasing when the PD has steps
        if cuts[i] <= cuts[i - 1]:
            cuts[i] = cuts[i - 1] + 1e-6
    return cuts


def run_tier_preview(engine_output: Path, cal_path: Path | None, cutoff_sets: list[list[float]], out_csv: Path | None,
                     share_sets: list[list[float]] | None = None) -> None:
    f = pd.read_csv(engine_output, low_memory=False)
    if "cal_pd" not in f.columns:
        print(f"{engine_output} has no cal_pd column")
        return
    cal = pd.read_csv(cal_path) if cal_path is not None and cal_path.exists() else None
    pdv = pd.to_numeric(f["cal_pd"], errors="coerce")
    print(f"\n=== Tier preview on {engine_output} ({int(pdv.notna().sum()):,} agents with cal_pd) ===")
    qs = [0.10, 0.25, 0.50, 0.75, 0.90]
    print("  agent PD percentiles: " + "  ".join(f"p{int(q * 100)}={pdv.quantile(q):.3f}" for q in qs))
    if cal_path is not None:
        print(f"  calibration map: {cal_path.resolve()}  ({'found' if cal is not None else 'NOT FOUND'})")
    if cal is None:
        print("  (no calibration map found: loan-level columns left out)")
    elif not {"model", "pd", "n", "bad_rate"} <= set(cal.columns):
        print(f"  (calibration map has columns {list(cal.columns)}; needs model, pd, n, bad_rate -- loan-level columns left out)")
    cutoff_sets = list(cutoff_sets)
    for shares in share_sets or []:
        cuts = cutoffs_for_shares(pdv, shares)
        print("  target agent shares " + " / ".join(f"{x:.0%}" for x in shares) + " -> cut-offs "
              + ", ".join(f"{c:.4f}" for c in cuts) + "  (tiers can differ slightly where many agents share one PD value)")
        cutoff_sets.append(cuts)
    frames = []
    for cuts in cutoff_sets:
        label = " / ".join(f"{c:.1%}" for c in cuts)
        print(f"\n--- cut-offs {label} ---")
        t = tier_preview(f, cuts, cal)
        print(_fmt(t))
        bc = by_category(f, cuts)
        if bc is not None:
            print("\n  % of each agent category in each tier:")
            print("  " + bc.to_string().replace("\n", "\n  "))
        frames.append(t.assign(cutoffs=label))
    if out_csv is not None:
        pd.concat(frames).to_csv(out_csv, index=False)
        print(f"\nWritten to {out_csv.resolve()}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Which model produced cal_pd? (read-only)")
    ap.add_argument("--engine-output", default="output/engine_test_output.csv")
    ap.add_argument("--artifacts-dir", default="pd_model/artifacts")
    ap.add_argument("--cutoffs", nargs="+", type=parse_cutoffs, metavar="C1,C2,C3",
                    help="Tier preview only: one or more sets of PD cut-offs, e.g. 0.02,0.05,0.10 0.03,0.12,0.30 "
                         "(percent also accepted: 3,12,30). Shows the share of scored agents per tier.")
    ap.add_argument("--target-shares", nargs="+", type=parse_shares, metavar="S1,S2,S3,S4",
                    help="Tier preview: agent shares wanted in tiers 1-4, e.g. 40,30,20,10 -- prints the PD cut-offs "
                         "that give them and previews those too.")
    ap.add_argument("--calibration-map", help="pd_calibration_map.csv for the loan-level columns "
                                              "(default: <artifacts-dir>/pd_calibration_map.csv)")
    ap.add_argument("--out", help="Tier preview: also write the tables to this CSV.")
    args = ap.parse_args(argv)
    art = Path(args.artifacts_dir)
    if args.cutoffs or args.target_shares:
        cal = Path(args.calibration_map) if args.calibration_map else art / "pd_calibration_map.csv"
        run_tier_preview(Path(args.engine_output), cal, args.cutoffs or [], Path(args.out) if args.out else None,
                         args.target_shares)
        return 0
    inspect_artifacts(art)
    inspect_calibration(art)
    inspect_thin_path(art)
    out = Path(args.engine_output)
    if out.exists():
        inspect_output(out)
    else:
        print(f"\nengine output not found: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
