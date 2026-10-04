#!/usr/bin/env bash
# Linux equivalent of run.bat -- same arguments, same behavior. See run.bat
# for the Windows version (kept for local dev on Windows machines).
set -u

python3 run_credit_risk_pipeline.py \
    --transaction-file data/mfs_daily_agent_mart_20260731.csv \
    --loan-file data/loan_summary.csv \
    --borrower-file data/borrower_history.csv \
    --loan-history-file data/loan_history_snapshot_20260619.csv \
    --snapshot-date 20260731 \
    --artifacts-dir pd_model/artifacts/ \
    --scorecard-path scorecards/capacity_scorecard_v1.json \
    --output output/engine_test_output.csv

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: run_credit_risk_pipeline.py failed with exit code $status"
    exit "$status"
fi

echo
echo "Pipeline complete. Output written to output/engine_test_output.csv"
