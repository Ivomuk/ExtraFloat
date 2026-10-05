#!/usr/bin/env bash
# Linux equivalent of test_axis2_pd_interaction.bat -- same behavior.
set -u

python3 scripts/test_axis2_pd_interaction.py "$@"

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: test_axis2_pd_interaction.py failed with exit code $status"
    exit "$status"
fi
