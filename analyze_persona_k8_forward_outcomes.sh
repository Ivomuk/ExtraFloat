#!/usr/bin/env bash
# Linux equivalent of analyze_persona_k8_forward_outcomes.bat -- same
# behavior. See analyze_persona_k8_forward_outcomes.bat for the Windows
# version (kept for local dev on Windows machines).
set -u

if [ -z "${1:-}" ]; then
    echo "Usage: analyze_persona_k8_forward_outcomes.sh <forward_outcomes_csv>"
    echo "  e.g.: analyze_persona_k8_forward_outcomes.sh data/persona_k8_forward_outcomes.csv"
    echo "  (export data/persona_k8_forward_outcomes_query.sql's result to that path first)"
    exit 1
fi

python3 scripts/analyze_persona_k8_forward_outcomes.py --forward-outcomes-file "$1"

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: analyze_persona_k8_forward_outcomes.py failed with exit code $status"
    exit "$status"
fi

echo
echo "Complete. See segmentation_outputs/persona_k8_profile/persona_k8_forward_outcomes_summary.csv"
