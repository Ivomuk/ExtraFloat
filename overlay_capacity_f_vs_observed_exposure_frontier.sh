#!/usr/bin/env bash
# Linux equivalent of overlay_capacity_f_vs_observed_exposure_frontier.bat -- same behavior.
set -u

python3 scripts/overlay_capacity_f_vs_observed_exposure_frontier.py "$@"

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: overlay_capacity_f_vs_observed_exposure_frontier.py failed with exit code $status"
    exit "$status"
fi
