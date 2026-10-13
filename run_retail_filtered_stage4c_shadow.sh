#!/usr/bin/env bash
# Linux equivalent of run_retail_filtered_stage4c_shadow.bat -- same 6
# steps, same arguments, same behavior. See that file for the Windows
# version and the full explanation of what Stage 4C shadow logging adds
# over run_retail_filtered.sh.
#
# Real artifact versions confirmed directly from the user's own files:
#   - pd_isotonic_calibrated_risk_metadata.json -> version "iso_v1_2026_10_06"
#     (C3 shadow calibration artifact -- used below for
#     --stage-4c-c3-shadow-artifact-version)
#   - capacity_artifacts/capacity_score_weights.json -> version
#     "capacity_weights_v1", recommended_weights 0.5/0.5 -- confirmed to
#     match derive_capacity_score_weights.py's real output schema exactly.
#
# The C3 shadow multiplier's artifact (pd_isotonic_calibrated_risk.joblib
# + its metadata) was previously hardwired to the same directory as
# --artifacts-dir in run_credit_risk_pipeline.py -- the new, independent
# --shadow-artifacts-dir flag below removes that coupling. CONFIRMED
# against the real files on the user's machine (a dir listing of both
# pd_model/artifacts/ and capacity_artifacts/): the real
# pd_isotonic_calibrated_risk.joblib + its metadata actually live in
# pd_model/artifacts/ (the SAME directory as --artifacts-dir) --
# capacity_artifacts/ holds only capacity_score_weights.json. An earlier
# confirmation that the C3 artifact lived in capacity_artifacts/ was
# wrong (the same mislabeling mix-up as the pasted-JSON confusion
# earlier in this workstream). --shadow-artifacts-dir is set to
# pd_model/artifacts/ below to match reality; if the artifact is later
# physically moved to capacity_artifacts/, update this flag to match.
#
# --stage-4c-c3-policy-version is left at "c3_v1" (the SHADOW_POLICY_
# VERSION constant in extrafloat_shadow_risk_multiplier.py) since the
# calibration metadata above only tracks the calibration version, not a
# separate policy version -- change it if a different convention is
# actually in use.
#
# --stage-4c-capacity-weights-file below assumes capacity_artifacts/
# capacity_score_weights.json sits at that path relative to wherever
# this script is run from (same relative-path convention as every other
# --*-file argument in this script) -- adjust if the real path differs.
set -u

echo "=== Step 1/6: Auditing retail filter against activity signal ==="
python3 scripts/audit_retail_filter_via_activity.py \
    --transaction-file data/mfs_daily_agent_mart_20260731.csv

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: audit_retail_filter_via_activity.py exited with code $status."
    echo "Exit code 2 means profile classification flags need human review before"
    echo "proceeding -- see retail_filter_activity_audit.csv. Any other non-zero"
    echo "code is a script error. Stopping before any filtering or scoring runs."
    exit "$status"
fi

echo
echo "=== Step 2/6: Classifying agents and filtering the transaction file ==="
python3 scripts/apply_retail_agent_filter.py \
    --transaction-file data/mfs_daily_agent_mart_20260731.csv \
    --out-retail retail_agents_filtered.csv \
    --out-excluded retail_agents_excluded.csv \
    --out-excluded-with-commission retail_agents_excluded_with_commission.csv

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: apply_retail_agent_filter.py failed with exit code $status"
    exit "$status"
fi

echo
echo "=== Step 3/6: Filtering the borrower file to the same retail-agent set ==="
python3 scripts/filter_borrower_file_by_retail_agents.py \
    --retail-agents-file retail_agents_filtered.csv \
    --borrower-file data/borrower_history.csv \
    --out borrower_history_retail_filtered.csv

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: filter_borrower_file_by_retail_agents.py failed with exit code $status"
    exit "$status"
fi

echo
echo "=== Step 4/6: Filtering the loan-summary file to the same retail-agent set ==="
python3 scripts/filter_borrower_file_by_retail_agents.py \
    --retail-agents-file retail_agents_filtered.csv \
    --borrower-file data/loan_summary.csv \
    --out data/loan_summary_retail_filtered.csv \
    --label "Loan-summary file"

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: filter_borrower_file_by_retail_agents.py failed with exit code $status"
    exit "$status"
fi

echo
echo "=== Step 5/6: Filtering the loan-history-snapshot file to the same retail-agent set ==="
python3 scripts/filter_borrower_file_by_retail_agents.py \
    --retail-agents-file retail_agents_filtered.csv \
    --borrower-file data/loan_history_snapshot_20260817.csv \
    --out data/loan_history_snapshot_20260817_retail_filtered.csv \
    --label "Loan-history-snapshot file"

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: filter_borrower_file_by_retail_agents.py failed with exit code $status"
    exit "$status"
fi

echo
echo "=== Step 6/6: Running the credit risk pipeline WITH Stage 4C shadow logging enabled ==="
echo "NOTE: --allow-provisional-scorecard is set since capacity_scorecard_v1.json"
echo "has not been through human sign-off yet -- this run is for pipeline"
echo "testing only, not a business-approved scoring run."
echo "Stage 4C output (Decision Log CSV + run manifest) will be written to"
echo "output/stage_4c_shadow/ -- inspect that manifest's shadow_logging_status"
echo "before treating this run as a clean pass."
python3 run_credit_risk_pipeline.py \
    --transaction-file retail_agents_filtered.csv \
    --loan-file data/loan_summary_retail_filtered.csv \
    --borrower-file borrower_history_retail_filtered.csv \
    --loan-history-file data/loan_history_snapshot_20260817_retail_filtered.csv \
    --snapshot-date 20260817 \
    --artifacts-dir pd_model/artifacts/ \
    --shadow-artifacts-dir pd_model/artifacts/ \
    --scorecard-path scorecards/capacity_scorecard_v1.json \
    --allow-provisional-scorecard \
    --keep-intermediate \
    --output output/engine_test_output_stage4c_shadow.csv \
    --stage-4c-logging-enabled \
    --stage-4c-output-dir output/stage_4c_shadow \
    --stage-4c-capacity-weights-file capacity_artifacts/capacity_score_weights.json \
    --stage-4c-c3-shadow-artifact-version iso_v1_2026_10_06 \
    --stage-4c-c3-policy-version c3_v1 \
    --stage-4c-fundamentals-source-version mfs_daily_agent_mart_20260731 > engine_log_stage4c_shadow.txt 2>&1

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: run_credit_risk_pipeline.py failed with exit code $status"
    echo
    echo "Last 20 lines of engine_log_stage4c_shadow.txt:"
    echo "----------------------------------------"
    tail -n 20 engine_log_stage4c_shadow.txt
    echo "----------------------------------------"
    echo "Full log: engine_log_stage4c_shadow.txt"
    exit "$status"
fi

echo
echo "Pipeline complete (retail-agents-only, every input file filtered, Stage 4C shadow logging enabled)."
echo "Live output written to output/engine_test_output_stage4c_shadow.csv"
echo "Stage 4C Decision Log + manifest written to output/stage_4c_shadow/"
echo "Full log: engine_log_stage4c_shadow.txt"
