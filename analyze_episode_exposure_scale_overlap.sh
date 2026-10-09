#!/usr/bin/env bash
# Linux equivalent of analyze_episode_exposure_scale_overlap.bat -- same behavior.
set -u

python3 scripts/analyze_episode_exposure_scale_overlap.py "$@"

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: analyze_episode_exposure_scale_overlap.py failed with exit code $status"
    exit "$status"
fi
