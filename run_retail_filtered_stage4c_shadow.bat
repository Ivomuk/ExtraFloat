@echo off

rem Identical to run_retail_filtered.bat (same 6-step retail-agent filter
rem pipeline), EXCEPT Step 6 additionally enables Stage 4C shadow Decision
rem Log logging, writing to its own dedicated output directory. This is
rem the controlled real-data Stage 4C shadow run -- it makes EXACTLY the
rem same live lending decisions as run_retail_filtered.bat; Stage 4C only
rem observes and logs, never feeds back into assigned_limit.
rem
rem BEFORE RUNNING: confirm your real C3 shadow calibration artifact
rem (pd_isotonic_calibrated_risk.joblib + its metadata JSON) is actually
rem present inside --artifacts-dir (pd_model/artifacts/, same directory
rem the PD model artifacts already live in -- compute_shadow_risk_
rem multiplier reads shadow_artifacts_dir=artifacts_dir, the SAME path).
rem If it is not there, shadow_multiplier_*/l_full_*/disagreement_* will
rem come back NULL for every row -- exactly as in the artifact-less
rem sandbox demo -- and that is the first thing to check if it happens
rem here, per the acceptance criteria already agreed.
rem
rem Fill in the two --stage-4c-c3-*-version values below from your real
rem artifact metadata (open pd_model/artifacts/pd_isotonic_calibrated_
rem risk_metadata.json and use its own "version" field for
rem --stage-4c-c3-shadow-artifact-version; use whatever your own c3
rem policy-version convention is for --stage-4c-c3-policy-version, e.g.
rem the SHADOW_POLICY_VERSION constant "c3_v1" if you have not
rem introduced a separate policy-version scheme). --stage-4c-capacity-
rem weights-file assumes you have already run scripts/derive_capacity_
rem score_weights.py -- if that artifact does not exist yet, remove that
rem line entirely and the run will fall back to the documented 0.5/0.5
rem default weights with a printed warning (fine for this first run).

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
    --scorecard-path scorecards/capacity_scorecard_v1.json ^
    --allow-provisional-scorecard ^
    --keep-intermediate ^
    --output output/engine_test_output_stage4c_shadow.csv ^
    --stage-4c-logging-enabled ^
    --stage-4c-output-dir output\stage_4c_shadow ^
    --stage-4c-capacity-weights-file capacity_artifacts\capacity_score_weights.json ^
    --stage-4c-c3-shadow-artifact-version FILL_IN_FROM_YOUR_METADATA_JSON ^
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
