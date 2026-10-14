"""
debug_stage_4c_shadow_merge.py
================================
TEMPORARY diagnostic script -- not part of the Stage 4C pipeline, not
covered by the one-way scripts/ layering convention's usual guarantees.
Reproduces build_decision_log_rows() against the REAL output files from a
real shadow run, with extra instrumentation printed at each step, to
pinpoint why shadow_multiplier_base/l_full_shadow_k0_05_base/
c3_shadow_artifact_version come back entirely null in the real Decision
Log despite the engine's own output CSV having shadow_multiplier_base
fully populated (confirmed separately).

Usage:
    python scripts\\debug_stage_4c_shadow_merge.py \\
        --engine-output-file output\\engine_test_output_stage4c_shadow.csv \\
        --transaction-file retail_agents_filtered.csv \\
        --run-id "2026-10-13T14:06:40.318057Z"

--engine-output-file MUST be the --keep-intermediate output (the shadow
run script already passes --keep-intermediate) -- it needs msisdn,
cal_pd, thin_file_flag, assigned_limit, policy_multiplier, capacity_cap,
final_decision_reason, shadow_multiplier_base, shadow_multiplier_conservative,
and ideally raw_model_score, all of which --keep-intermediate preserves.
--transaction-file is the SAME file the real shadow run passed as
--transaction-file (the raw agent_msisdn-keyed transaction file; retail
runs use retail_agents_filtered.csv).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from extrafloat.shadow_logging.stage_4c_decision_logger import (  # noqa: E402
    _norm_msisdn,
    attach_challenger_risk_signal,
    build_decision_log_rows,
)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--engine-output-file", required=True)
    p.add_argument("--transaction-file", required=True)
    p.add_argument("--run-id", required=True)
    args = p.parse_args(argv)

    print("=" * 80)
    print("Loading files...")
    result_df = pd.read_csv(args.engine_output_file, sep=",")
    df_agent_raw = pd.read_csv(args.transaction_file, sep=",")
    print(f"engine output rows: {len(result_df)}, columns: {len(result_df.columns)}")
    print(f"transaction file rows: {len(df_agent_raw)}, columns: {len(df_agent_raw.columns)}")

    print("=" * 80)
    print("Step 1: required columns present on engine output?")
    for col in ["msisdn", "cal_pd", "thin_file_flag", "shadow_multiplier_base",
                "shadow_multiplier_conservative", "assigned_limit",
                "policy_multiplier", "capacity_cap", "final_decision_reason"]:
        print(f"  {col:35s} present={col in result_df.columns}")

    print("=" * 80)
    print("Step 2: duplicate COLUMN NAMES in engine output? (would silently break")
    print("        'shadow_cols <= set(result.columns)' style checks)")
    dup_cols = result_df.columns[result_df.columns.duplicated()].tolist()
    print(f"  duplicate column names: {dup_cols if dup_cols else 'NONE'}")

    print("=" * 80)
    print("Step 3: msisdn dtype + duplicate VALUES in engine output")
    print(f"  dtype: {result_df['msisdn'].dtype}")
    print(f"  duplicated msisdn rows: {result_df['msisdn'].duplicated().sum()} / {len(result_df)}")
    print(f"  null msisdn rows: {result_df['msisdn'].isna().sum()}")
    print(f"  sample raw values: {result_df['msisdn'].head(3).tolist()}")

    print("=" * 80)
    print("Step 4: shadow_multiplier_base non-null BEFORE any Decision Log code runs")
    print(f"  non-null: {result_df['shadow_multiplier_base'].notna().sum()} / {len(result_df)}")

    print("=" * 80)
    print("Step 5: _norm_msisdn applied to engine output's own msisdn column")
    normed = _norm_msisdn(result_df["msisdn"].copy())
    print(f"  normed dtype: {normed.dtype}")
    print(f"  normed duplicated: {normed.duplicated().sum()}")
    print(f"  normed null/NA: {normed.isna().sum()}")
    print(f"  sample normed values: {normed.head(3).tolist()}")

    print("=" * 80)
    print("Step 6: self-join sanity check -- merging the engine output against")
    print("        itself on normalized msisdn (exactly what the real shadow")
    print("        merge does) -- should be ~100% match by construction")
    result = result_df.copy()
    result["msisdn"] = _norm_msisdn(result["msisdn"])
    base = result[["msisdn"]].drop_duplicates(subset=["msisdn"]).reset_index(drop=True)
    shadow_df = result[["msisdn", "shadow_multiplier_base", "shadow_multiplier_conservative"]] \
        .drop_duplicates(subset=["msisdn"])
    merged = base.merge(shadow_df, on="msisdn", how="left")
    print(f"  base rows: {len(base)}  shadow_df rows: {len(shadow_df)}  merged rows: {len(merged)}")
    print(f"  merged shadow_multiplier_base non-null: {merged['shadow_multiplier_base'].notna().sum()} / {len(merged)}")

    print("=" * 80)
    print("Step 7: attach_challenger_risk_signal on its own (this is what")
    print("        populated risk_route/cal_pd correctly in the real run)")
    risk = attach_challenger_risk_signal(result)
    print(f"  risk rows: {len(risk)}")
    print(f"  risk_route value counts:\n{risk['risk_route'].value_counts(dropna=False).to_string()}")

    print("=" * 80)
    print("Step 8: full build_decision_log_rows() end-to-end on REAL data")
    try:
        out = build_decision_log_rows(
            df_agent_raw=df_agent_raw,
            result_df=result_df,
            run_id=args.run_id,
            scoring_run_timestamp=args.run_id,
            capacity_weights={"float_activity_value_1m": 0.5, "commission": 0.5},
            capacity_weights_version="debug_v1",
            artifact_versions={
                "champion_model_version": "xgb",
                "pd_calibration_artifact_version": None,
                "c3_shadow_artifact_version": "iso_v1_2026_10_06",
                "c3_policy_version": "c3_v1",
                "fundamentals_source_version": "mart_v1",
            },
        )
        print(f"  rows: {len(out)}")
        print(f"  shadow_multiplier_base non-null: {out['shadow_multiplier_base'].notna().sum()} / {len(out)}")
        print(f"  l_full_shadow_k0_05_base non-null: {out['l_full_shadow_k0_05_base'].notna().sum()} / {len(out)}")
    except Exception as exc:
        print(f"  RAISED: {type(exc).__name__}: {exc}")

    print("=" * 80)
    print("DONE -- paste this entire output back.")


if __name__ == "__main__":
    sys.exit(main())
