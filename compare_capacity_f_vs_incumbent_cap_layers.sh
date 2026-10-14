#!/usr/bin/env bash
# Linux equivalent of compare_capacity_f_vs_incumbent_cap_layers.bat -- same behavior.
set -u

python3 scripts/compare_capacity_f_vs_incumbent_cap_layers.py "$@"

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: compare_capacity_f_vs_incumbent_cap_layers.py failed with exit code $status"
    exit "$status"
fi
