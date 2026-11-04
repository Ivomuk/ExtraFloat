#!/usr/bin/env bash
# Linux equivalent of summarize_vs_disbursed.bat -- same behavior.
set -u

python3 -m extrafloat.cap_proposal.summarize_vs_disbursed "$@"

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: summarize_vs_disbursed failed with exit code $status"
    exit "$status"
fi
