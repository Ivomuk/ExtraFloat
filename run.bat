@echo off
python run_credit_risk_pipeline.py ^
    --transaction-file data/mfs_daily_agent_mart_20260731.csv ^
    --loan-file data/loan_summary.csv ^
    --borrower-file data/borrower_history.csv ^
    --repayment-file data/snapshots_202608112017.csv ^
    --snapshot-date 20260619 ^
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
echo Pipeline complete. Output written to output/engine_test_output.csv
pause
