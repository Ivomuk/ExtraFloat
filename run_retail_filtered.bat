@echo off

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
echo === Step 6/6: Running the credit risk pipeline on the retail-only population ===
python run_credit_risk_pipeline.py ^
    --transaction-file retail_agents_filtered.csv ^
    --loan-file data\loan_summary_retail_filtered.csv ^
    --borrower-file borrower_history_retail_filtered.csv ^
    --loan-history-file data\loan_history_snapshot_20260817_retail_filtered.csv ^
    --snapshot-date 20260817 ^
    --artifacts-dir pd_model/artifacts/ ^
    --scorecard-path scorecards/capacity_scorecard_v1.json ^
    --output output/engine_test_output.csv

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: run_credit_risk_pipeline.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo Pipeline complete (retail-agents-only, every input file filtered).
echo Output written to output/engine_test_output.csv
pause
