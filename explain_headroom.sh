#!/usr/bin/env bash
# Linux equivalent of explain_headroom.bat -- same behavior.
set -u

python3 -m extrafloat.cap_proposal.explain_headroom "$@"

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: explain_headroom failed with exit code $status"
    exit "$status"
fi
