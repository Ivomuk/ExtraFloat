#!/usr/bin/env bash
# Linux equivalent of train_retail_filtered.bat -- same 6 steps, same
# arguments, same behavior. See train_retail_filtered.bat for the Windows
# version (kept for local dev on Windows machines).
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
    echo "code is a script error. Stopping before any filtering or training runs."
    exit "$status"
fi

echo
echo "=== Step 2/6: Classifying agents from the val-snapshot transaction file ==="
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
echo "=== Step 3/6: Filtering the loan-training file to the same retail-agent set ==="
python3 scripts/filter_borrower_file_by_retail_agents.py \
    --retail-agents-file retail_agents_filtered.csv \
    --borrower-file data/state_data_20260910.csv \
    --out data/state_data_20260910_retail_filtered.csv \
    --label "Loan-training file"

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: filter_borrower_file_by_retail_agents.py failed with exit code $status"
    exit "$status"
fi

echo
echo "=== Step 4/6: Filtering the train-snapshot file (May) to the same retail-agent set ==="
echo "NOTE: uses the SAME allowlist derived from the July snapshot (Step 2) --"
echo "one retail-agent definition applied across the whole run, consistent"
echo "with the loan-training-file filter above. If an agent's agent_profile"
echo "genuinely changed between May and July, this snapshot's classification"
echo "follows July's -- see the \"not found in\" counts below for how many"
echo "agents this affects."
python3 scripts/filter_borrower_file_by_retail_agents.py \
    --retail-agents-file retail_agents_filtered.csv \
    --borrower-file data/mfs_daily_agent_mart_20260531.csv \
    --out data/mfs_daily_agent_mart_20260531_retail_filtered.csv \
    --label "Train-snapshot file (May)"

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: filter_borrower_file_by_retail_agents.py failed with exit code $status"
    exit "$status"
fi

echo
echo "=== Step 5/6: Filtering the val-snapshot file (July) to the same retail-agent set ==="
python3 scripts/filter_borrower_file_by_retail_agents.py \
    --retail-agents-file retail_agents_filtered.csv \
    --borrower-file data/mfs_daily_agent_mart_20260731.csv \
    --out data/mfs_daily_agent_mart_20260731_retail_filtered.csv \
    --label "Val-snapshot file (July)"

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: filter_borrower_file_by_retail_agents.py failed with exit code $status"
    exit "$status"
fi

echo
echo "=== Step 6/6: Training the PD model on the retail-only population ==="
python3 -m pd_model.run_pipeline \
    --train-file          data/mfs_daily_agent_mart_20260531_retail_filtered.csv \
    --val-file             data/mfs_daily_agent_mart_20260731_retail_filtered.csv \
    --loan-training-file   data/state_data_20260910_retail_filtered.csv \
    --train-snapshot-date  20260531 \
    --val-snapshot-date    20260731 \
    --output-dir           pd_model/artifacts \
    --champion             xgb > training_log.txt 2>&1

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: pd_model training failed with exit code $status"
    echo
    echo "Last 20 lines of training_log.txt:"
    echo "----------------------------------------"
    tail -n 20 training_log.txt
    echo "----------------------------------------"
    echo "Full log: training_log.txt"
    exit "$status"
fi

echo
echo "Training complete (retail-agents-only). Artifacts written to pd_model/artifacts/"
echo "Full log: training_log.txt"
