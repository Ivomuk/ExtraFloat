#!/usr/bin/env bash
# Linux equivalent of train.bat -- same arguments, same behavior. See
# train.bat for the Windows version (kept for local dev on Windows machines).
set -u

python3 -m pd_model.run_pipeline \
    --train-file          data/mfs_daily_agent_mart_20260531.csv \
    --val-file             data/mfs_daily_agent_mart_20260731.csv \
    --loan-training-file   data/state_data_202608131214.csv \
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
echo "Training complete. Artifacts written to pd_model/artifacts/"
echo "Full log: training_log.txt"
