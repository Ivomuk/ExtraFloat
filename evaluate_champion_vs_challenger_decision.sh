#!/usr/bin/env bash
# Linux equivalent of evaluate_champion_vs_challenger_decision.bat -- same behavior.
set -u

python3 scripts/evaluate_champion_vs_challenger_decision.py "$@"

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: evaluate_champion_vs_challenger_decision.py failed with exit code $status"
    exit "$status"
fi
