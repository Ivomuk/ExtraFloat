#!/usr/bin/env bash
# Linux equivalent of run_retail_filtered.bat -- same 6 steps, same
# arguments, same behavior. See run_retail_filtered.bat for the Windows
# version (kept for local dev on Windows machines).
#
# This is also step 1 of persona generation on Linux: it produces
# borrower_history_retail_filtered.csv and output/engine_test_output.csv,
# both required by profile_persona_k8.sh. Run this first, then
# profile_persona_k8.sh.
set -u

echo "=== Step 1/6: Auditing retail filter against activity signal ==="
python3 scripts/audit_retail_filter_via_activity.py \
    --transaction-file data/mfs_daily_agent_mart_20260731.csv

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: audit_retail_filter_via_activity.py exited with code $status."
    echo "Exit code 2 means profile classification flags need human review before"
    echo "proceeding -- see retail_filter_activity_audit.csv. Any other non-zero"
    echo "code is a script error. Stopping before any filtering or scoring runs."
    exit "$status"
fi

echo
echo "=== Step 2/6: Classifying agents and filtering the transaction file ==="
python3 scripts/apply_retail_agent_filter.py \
    --transaction-file data/mfs_daily_agent_mart_20260731.csv \
    --out-retail retail_agents_filtered.csv \
    --out-excluded retail_agents_excluded.csv \
    --out-excluded-with-commission retail_agents_excluded_with_commission.csv

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: apply_retail_agent_filter.py failed with exit code $status"
    exit "$status"
fi

echo
echo "=== Step 3/6: Filtering the borrower file to the same retail-agent set ==="
python3 scripts/filter_borrower_file_by_retail_agents.py \
    --retail-agents-file retail_agents_filtered.csv \
    --borrower-file data/borrower_history.csv \
    --out borrower_history_retail_filtered.csv

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: filter_borrower_file_by_retail_agents.py failed with exit code $status"
    exit "$status"
fi

echo
echo "=== Step 4/6: Filtering the loan-summary file to the same retail-agent set ==="
python3 scripts/filter_borrower_file_by_retail_agents.py \
    --retail-agents-file retail_agents_filtered.csv \
    --borrower-file data/loan_summary.csv \
    --out data/loan_summary_retail_filtered.csv \
    --label "Loan-summary file"

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: filter_borrower_file_by_retail_agents.py failed with exit code $status"
    exit "$status"
fi

echo
echo "=== Step 5/6: Filtering the loan-history-snapshot file to the same retail-agent set ==="
python3 scripts/filter_borrower_file_by_retail_agents.py \
    --retail-agents-file retail_agents_filtered.csv \
    --borrower-file data/loan_history_snapshot_20260817.csv \
    --out data/loan_history_snapshot_20260817_retail_filtered.csv \
    --label "Loan-history-snapshot file"

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: filter_borrower_file_by_retail_agents.py failed with exit code $status"
    exit "$status"
fi

echo
echo "=== Step 6/6: Running the credit risk pipeline on the retail-only population ==="
echo "NOTE: --allow-provisional-scorecard is set since capacity_scorecard_v1.json"
echo "has not been through human sign-off yet -- this run is for pipeline"
echo "testing only, not a business-approved scoring run."
python3 run_credit_risk_pipeline.py \
    --transaction-file retail_agents_filtered.csv \
    --loan-file data/loan_summary_retail_filtered.csv \
    --borrower-file borrower_history_retail_filtered.csv \
    --loan-history-file data/loan_history_snapshot_20260817_retail_filtered.csv \
    --snapshot-date 20260817 \
    --artifacts-dir pd_model/artifacts/ \
    --scorecard-path scorecards/capacity_scorecard_v1.json \
    --allow-provisional-scorecard \
    --output output/engine_test_output.csv > engine_log.txt 2>&1

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: run_credit_risk_pipeline.py failed with exit code $status"
    echo
    echo "Last 20 lines of engine_log.txt:"
    echo "----------------------------------------"
    tail -n 20 engine_log.txt
    echo "----------------------------------------"
    echo "Full log: engine_log.txt"
    exit "$status"
fi

echo
echo "Pipeline complete (retail-agents-only, every input file filtered)."
echo "Output written to output/engine_test_output.csv"
echo "Full log: engine_log.txt"
