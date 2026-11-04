#!/usr/bin/env bash
# Linux equivalent of compare_scenarios.bat -- same behavior.
set -u

python3 -m extrafloat.cap_proposal.compare_scenarios "$@"

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: compare_scenarios failed with exit code $status"
    exit "$status"
fi
