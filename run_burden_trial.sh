#!/usr/bin/env bash
# Linux equivalent of run_burden_trial.bat -- same behavior.
set -u

python3 -m extrafloat.cap_proposal.run_burden_trial "$@"

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: run_burden_trial failed with exit code $status"
    exit "$status"
fi
