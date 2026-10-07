#!/usr/bin/env bash
# Linux equivalent of export_champion_vs_challenger_limits.bat -- same behavior.
set -u

python3 scripts/export_champion_vs_challenger_limits.py "$@"

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: export_champion_vs_challenger_limits.py failed with exit code $status"
    exit "$status"
fi
