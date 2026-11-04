#!/usr/bin/env bash
# Linux equivalent of validate_burden_history.bat -- same behavior.
set -u

python3 -m extrafloat.cap_proposal.validate_burden_history "$@"

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: validate_burden_history failed with exit code $status"
    exit "$status"
fi
