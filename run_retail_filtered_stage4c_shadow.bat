@echo off

rem Identical to run_retail_filtered.bat (same 6-step retail-agent filter
rem pipeline), EXCEPT Step 6 additionally enables Stage 4C shadow Decision
rem Log logging, writing to its own dedicated output directory. This is
rem the controlled real-data Stage 4C shadow run -- it makes EXACTLY the
rem same live lending decisions as run_retail_filtered.bat; Stage 4C only
rem observes and logs, never feeds back into assigned_limit.
rem
rem Real artifact versions confirmed directly from the user's own files:
rem   - pd_isotonic_calibrated_risk_metadata.json -> version "iso_v1_2026_10_06"
rem     (C3 shadow calibration artifact -- used below for
rem     --stage-4c-c3-shadow-artifact-version)
rem   - capacity_artifacts\capacity_score_weights.json -> version
rem     "capacity_weights_v1", recommended_weights 0.5/0.5 -- confirmed to
rem     match derive_capacity_score_weights.py's real output schema exactly.
rem
rem The C3 shadow multiplier's artifact (pd_isotonic_calibrated_risk.joblib
rem + its metadata) was previously hardwired to the same directory as
rem --artifacts-dir in run_credit_risk_pipeline.py -- the new, independent
rem --shadow-artifacts-dir flag below removes that coupling. CONFIRMED
rem against the real files on the user's machine (a dir listing of both
rem pd_model\artifacts\ and capacity_artifacts\): the real
rem pd_isotonic_calibrated_risk.joblib + its metadata actually live in
rem pd_model\artifacts\ (the SAME directory as --artifacts-dir) --
rem capacity_artifacts\ holds only capacity_score_weights.json. An
rem earlier confirmation that the C3 artifact lived in capacity_
rem artifacts\ was wrong (the same mislabeling mix-up as the pasted-JSON
rem confusion earlier in this workstream). --shadow-artifacts-dir is set
rem to pd_model/artifacts/ below to match reality; if the artifact is
rem later physically moved to capacity_artifacts\, update this flag to
rem match.
rem
rem --stage-4c-c3-policy-version is left at "c3_v1" (the SHADOW_POLICY_
rem VERSION constant in extrafloat_shadow_risk_multiplier.py) since the
rem calibration metadata above only tracks the calibration version, not
rem a separate policy version -- change it if a different convention is
rem actually in use.
rem
rem --stage-4c-capacity-weights-file below assumes capacity_artifacts\
rem capacity_score_weights.json sits at that path relative to wherever
rem this script is run from (same relative-path convention as every
rem other --*-file argument in this script) -- adjust if the real path
rem differs on this machine.

echo === Step 1/6: Auditing retail filter against activity signal ===
python scripts\audit_retail_filter_via_activity.py ^
    --transaction-file data\mfs_daily_agent_mart_20260731.csv

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: audit_retail_filter_via_activity.py exited with code %ERRORLEVEL%.
    echo Exit code 2 means profile classification flags need human review before
    echo proceeding -- see retail_filter_activity_audit.csv. Any other non-zero
    echo code is a script error. Stopping before any filtering or scoring runs.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo === Step 2/6: Classifying agents and filtering the transaction file ===
python scripts\apply_retail_agent_filter.py ^
    --transaction-file data\mfs_daily_agent_mart_20260731.csv ^
    --out-retail retail_agents_filtered.csv ^
    --out-excluded retail_agents_excluded.csv ^
    --out-excluded-with-commission retail_agents_excluded_with_commission.csv

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: apply_retail_agent_filter.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo === Step 3/6: Filtering the borrower file to the same retail-agent set ===
python scripts\filter_borrower_file_by_retail_agents.py ^
    --retail-agents-file retail_agents_filtered.csv ^
    --borrower-file data\borrower_history.csv ^
    --out borrower_history_retail_filtered.csv

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: filter_borrower_file_by_retail_agents.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo === Step 4/6: Filtering the loan-summary file to the same retail-agent set ===
python scripts\filter_borrower_file_by_retail_agents.py ^
    --retail-agents-file retail_agents_filtered.csv ^
    --borrower-file data\loan_summary.csv ^
    --out data\loan_summary_retail_filtered.csv ^
    --label "Loan-summary file"

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: filter_borrower_file_by_retail_agents.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo === Step 5/6: Filtering the loan-history-snapshot file to the same retail-agent set ===
python scripts\filter_borrower_file_by_retail_agents.py ^
    --retail-agents-file retail_agents_filtered.csv ^
    --borrower-file data\loan_history_snapshot_20260817.csv ^
    --out data\loan_history_snapshot_20260817_retail_filtered.csv ^
    --label "Loan-history-snapshot file"

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: filter_borrower_file_by_retail_agents.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo === Step 6/6: Running the credit risk pipeline WITH Stage 4C shadow logging enabled ===
echo NOTE: --allow-provisional-scorecard is set since capacity_scorecard_v1.json
echo has not been through human sign-off yet -- this run is for pipeline
echo testing only, not a business-approved scoring run.
echo Stage 4C output (Decision Log CSV + run manifest) will be written to
echo output\stage_4c_shadow\ -- inspect that manifest's shadow_logging_status
echo before treating this run as a clean pass.
python run_credit_risk_pipeline.py ^
    --transaction-file retail_agents_filtered.csv ^
    --loan-file data\loan_summary_retail_filtered.csv ^
    --borrower-file borrower_history_retail_filtered.csv ^
    --loan-history-file data\loan_history_snapshot_20260817_retail_filtered.csv ^
    --snapshot-date 20260817 ^
    --artifacts-dir pd_model/artifacts/ ^
    --shadow-artifacts-dir pd_model/artifacts/ ^
    --scorecard-path scorecards/capacity_scorecard_v1.json ^
    --allow-provisional-scorecard ^
    --keep-intermediate ^
    --output output/engine_test_output_stage4c_shadow.csv ^
    --stage-4c-logging-enabled ^
    --stage-4c-output-dir output\stage_4c_shadow ^
    --stage-4c-capacity-weights-file capacity_artifacts\capacity_score_weights.json ^
    --stage-4c-c3-shadow-artifact-version iso_v1_2026_10_06 ^
    --stage-4c-c3-policy-version c3_v1 ^
    --stage-4c-fundamentals-source-version mfs_daily_agent_mart_20260731 > engine_log_stage4c_shadow.txt 2>&1

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: run_credit_risk_pipeline.py failed with exit code %ERRORLEVEL%
    echo.
    echo Last 20 lines of engine_log_stage4c_shadow.txt:
    echo ----------------------------------------
    powershell -Command "Get-Content engine_log_stage4c_shadow.txt -Tail 20"
    echo ----------------------------------------
    echo Full log: engine_log_stage4c_shadow.txt
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo Pipeline complete (retail-agents-only, every input file filtered, Stage 4C shadow logging enabled).
echo Live output written to output/engine_test_output_stage4c_shadow.csv
echo Stage 4C Decision Log + manifest written to output\stage_4c_shadow\
echo Full log: engine_log_stage4c_shadow.txt
pause
