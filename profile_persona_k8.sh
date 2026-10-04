#!/usr/bin/env bash
# Linux equivalent of profile_persona_k8.bat -- same behavior. See
# profile_persona_k8.bat for the Windows version (kept for local dev on
# Windows machines).
#
# Requires borrower_history_retail_filtered.csv and (optionally, for the
# held-out risk/limit columns in k8_validation_crosstabs.xlsx)
# output/engine_test_output.csv -- both produced by run_retail_filtered.sh.
# Run that first if those files don't exist yet.
set -u

python3 scripts/profile_persona_k8.py

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: profile_persona_k8.py failed with exit code $status"
    exit "$status"
fi

echo
echo "Complete. See segmentation_outputs/persona_k8_profile/ for the 4 artifacts."
