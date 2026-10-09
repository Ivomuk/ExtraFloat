#!/usr/bin/env bash
# Linux equivalent of analyze_episode_exposure_escalation_matrix.bat -- same behavior.
set -u

python3 scripts/analyze_episode_exposure_escalation_matrix.py "$@"

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: analyze_episode_exposure_escalation_matrix.py failed with exit code $status"
    exit "$status"
fi
