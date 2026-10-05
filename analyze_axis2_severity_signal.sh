#!/usr/bin/env bash
# Linux equivalent of analyze_axis2_severity_signal.bat -- same behavior.
set -u

python3 scripts/analyze_axis2_severity_signal.py "$@"

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: analyze_axis2_severity_signal.py failed with exit code $status"
    exit "$status"
fi
