"""
Compares actual August 1-14 disbursements (raw devdata.fin_log transaction
export -- RESERVATION rows where instruct_from_sp = 'XTRAFLOAT-AGENT', i.e.
XtraFloat sending money out to an agent) against the PD engine's own
assigned_limit for each recipient agent, to check whether disbursements
actually respected the model's risk-based limit.

Per-transaction comparison (not cumulative): each individual disbursement
is checked against the agent's assigned_limit on its own.

Expects the disbursements file to have (at minimum) instruct_to_fro_msisdn
(recipient agent), instruct_amount (disbursed amount), and tbl_dt (date,
YYYYMMDD). Column names match the devdata.fin_log export query that
produced it.

Usage:
    python scripts\\check_disbursements_vs_assigned_limit.py ^
        --disbursements-file data\\fin_log_202609181256 ^
        --engine-output output\\engine_test_output.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

REQUIRED_DISB_COLS = ["instruct_to_fro_msisdn", "instruct_amount", "tbl_dt"]


def _normalize_msisdn(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"[^0-9]", "", regex=True).replace("", pd.NA)


def _read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in (".csv", "") or path.suffix.lower() not in (".tsv", ".txt"):
        try:
            return pd.read_csv(path)
        except Exception:
            pass
    for sep in ["\t", "|", ";"]:
        try:
            df = pd.read_csv(path, sep=sep)
            if df.shape[1] > 1:
                return df
        except Exception:
            continue
    sys.exit(f"ERROR: could not parse {path} as CSV/TSV. Check its delimiter and re-run with "
              f"the right pandas.read_csv args, or tell me the actual format.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--disbursements-file", required=True)
    ap.add_argument("--engine-output", default="output/engine_test_output.csv")
    ap.add_argument("--out-prefix", default="disbursements_vs_limit")
    args = ap.parse_args()

    disb_path = Path(args.disbursements_file)
    if not disb_path.exists():
        sys.exit(f"ERROR: disbursements file not found: {disb_path}")
    eng_path = Path(args.engine_output)
    if not eng_path.exists():
        sys.exit(f"ERROR: engine output file not found: {eng_path}")

    disb = _read_any(disb_path)
    disb.columns = [c.strip() for c in disb.columns]
    missing = [c for c in REQUIRED_DISB_COLS if c not in disb.columns]
    if missing:
        sys.exit(f"ERROR: {disb_path} is missing required column(s): {missing}. "
                  f"Columns present: {list(disb.columns)}")

    eng = pd.read_csv(eng_path)
    eng_msisdn_col = "msisdn" if "msisdn" in eng.columns else ("agent_msisdn" if "agent_msisdn" in eng.columns else None)
    if eng_msisdn_col is None or "assigned_limit" not in eng.columns:
        sys.exit(f"ERROR: {eng_path} needs a msisdn/agent_msisdn column and assigned_limit. "
                  f"Columns present: {list(eng.columns)}")

    disb["_key"] = _normalize_msisdn(disb["instruct_to_fro_msisdn"])
    disb["instruct_amount"] = pd.to_numeric(disb["instruct_amount"], errors="coerce")
    n_disb = len(disb)
    print(f"Disbursement transactions (Aug 1-14): {n_disb:,}")
    print(f"Distinct recipient msisdns: {disb['_key'].nunique():,}\n")

    eng["_key"] = _normalize_msisdn(eng[eng_msisdn_col])
    eng_cols = ["_key", "assigned_limit"]
    for extra in ["cal_pd", "risk_tier", "pd_decile", "final_decision_reason"]:
        if extra in eng.columns:
            eng_cols.append(extra)
    eng_small = eng[eng_cols].drop_duplicates(subset="_key")

    merged = disb.merge(eng_small, on="_key", how="left")
    n_matched = merged["assigned_limit"].notna().sum()
    print(f"Matched to an assigned_limit in {eng_path.name}: {n_matched:,} / {n_disb:,} "
          f"({n_matched / n_disb:.1%})")
    n_unmatched = n_disb - n_matched
    if n_unmatched:
        print(f"NOTE: {n_unmatched:,} disbursement(s) have no matching agent in the engine "
              f"output at all -- these agents were never scored/limited by this model run, "
              f"so they can't be a limit violation in the sense checked below.\n")

    scored = merged[merged["assigned_limit"].notna()].copy()
    scored["exceeds_limit"] = scored["instruct_amount"] > scored["assigned_limit"]
    scored["excess_amount"] = (scored["instruct_amount"] - scored["assigned_limit"]).clip(lower=0)
    scored["pct_of_limit"] = scored["instruct_amount"] / scored["assigned_limit"].replace(0, pd.NA)

    n_exceed = int(scored["exceeds_limit"].sum())
    n_zero_limit_disbursed = int(((scored["assigned_limit"] == 0) & (scored["instruct_amount"] > 0)).sum())

    print("=" * 78)
    print("PER-TRANSACTION comparison: disbursed amount vs. assigned_limit")
    print("=" * 78)
    print(f"Transactions exceeding assigned_limit: {n_exceed:,} / {len(scored):,} "
          f"({n_exceed / len(scored):.1%})")
    print(f"  of which, assigned_limit == 0 but still disbursed: {n_zero_limit_disbursed:,}")
    if n_exceed:
        exceed_rows = scored[scored["exceeds_limit"]]
        print(f"\nExcess amount (disbursed - limit), among violations:")
        print(f"  min={exceed_rows['excess_amount'].min():,.0f}  median={exceed_rows['excess_amount'].median():,.0f}  "
              f"mean={exceed_rows['excess_amount'].mean():,.0f}  max={exceed_rows['excess_amount'].max():,.0f}")
        print(f"\npct_of_limit distribution among violations (>100% = over limit):")
        print(f"  median={exceed_rows['pct_of_limit'].median():.1%}  "
              f"p90={exceed_rows['pct_of_limit'].quantile(0.9):.1%}  max={exceed_rows['pct_of_limit'].max():.1%}")

    print(f"\npct_of_limit distribution, ALL matched transactions (not just violations):")
    print(f"  median={scored['pct_of_limit'].median():.1%}  p75={scored['pct_of_limit'].quantile(0.75):.1%}  "
          f"p90={scored['pct_of_limit'].quantile(0.9):.1%}  max={scored['pct_of_limit'].max():.1%}")

    out_cols = ["_key", "tbl_dt", "instruct_amount", "assigned_limit", "exceeds_limit",
                "excess_amount", "pct_of_limit"] + [c for c in ["cal_pd", "risk_tier", "pd_decile"] if c in scored.columns]
    out_path = f"{args.out_prefix}_per_transaction.csv"
    scored[out_cols].rename(columns={"_key": "msisdn"}).sort_values("excess_amount", ascending=False).to_csv(out_path, index=False)
    print(f"\nPer-transaction detail written: {out_path}")


if __name__ == "__main__":
    main()
