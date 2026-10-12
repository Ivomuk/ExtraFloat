#!/usr/bin/env bash
# Linux equivalent of diagnose_capacity_score_tail_composition.bat -- same behavior.
set -u

python3 scripts/diagnose_capacity_score_tail_composition.py "$@"

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: diagnose_capacity_score_tail_composition.py failed with exit code $status"
    exit "$status"
fi
