#!/usr/bin/env bash
# Linux equivalent of compare_limits_vs_disbursed.bat -- same behavior.
set -u

python3 -m extrafloat.cap_proposal.compare_limits_vs_disbursed "$@"

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: compare_limits_vs_disbursed failed with exit code $status"
    exit "$status"
fi
