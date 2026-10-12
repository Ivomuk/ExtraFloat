#!/usr/bin/env bash
# Linux equivalent of score_retrospective_current_model.bat -- same behavior.
set -u

python3 scripts/score_retrospective_current_model.py "$@"

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: score_retrospective_current_model.py failed with exit code $status"
    exit "$status"
fi
