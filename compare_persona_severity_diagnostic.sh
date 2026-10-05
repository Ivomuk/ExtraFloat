#!/usr/bin/env bash
# Linux equivalent of compare_persona_severity_diagnostic.bat -- same behavior.
set -u

python3 scripts/compare_persona_severity_diagnostic.py "$@"

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: compare_persona_severity_diagnostic.py failed with exit code $status"
    exit "$status"
fi
