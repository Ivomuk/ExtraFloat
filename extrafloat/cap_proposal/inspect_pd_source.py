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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Which model produced cal_pd? (read-only)")
    ap.add_argument("--engine-output", default="output/engine_test_output.csv")
    ap.add_argument("--artifacts-dir", default="pd_model/artifacts")
    args = ap.parse_args(argv)
    art = Path(args.artifacts_dir)
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
