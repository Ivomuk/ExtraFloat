#!/usr/bin/env bash
# Linux equivalent of build_retrospective_scoring_matrix.bat -- same behavior.
set -u

python3 scripts/build_retrospective_scoring_matrix.py "$@"

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: build_retrospective_scoring_matrix.py failed with exit code $status"
    exit "$status"
fi
