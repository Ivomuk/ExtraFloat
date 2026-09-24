@echo off
echo === Tracing whitelist agents through every retail-filtered pipeline stage ===
echo Not just the final scored output -- also raw source data, the retail-agent
echo filter itself, and each intermediate filtered file, so we can see exactly
echo WHERE a missing agent first drops out.

python scripts\check_whitelist_not_scored.py ^
    --whitelist-file data\whitelist_aug_20260804.csv ^
    --stage "raw_agent_mart=data\mfs_daily_agent_mart_20260731.csv" ^
    --stage "retail_filtered=retail_agents_filtered.csv" ^
    --stage "borrower_raw=data\borrower_history.csv" ^
    --stage "borrower_filtered=borrower_history_retail_filtered.csv" ^
    --stage "loan_summary_filtered=data\loan_summary_retail_filtered.csv" ^
    --stage "loan_history_filtered=data\loan_history_snapshot_20260817_retail_filtered.csv" ^
    --stage "scored_output=output\engine_test_output.csv" ^
    --final-stage scored_output ^
    --out-missing whitelist_agents_not_scored.csv

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: check_whitelist_not_scored.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo Complete. See whitelist_agents_not_scored.csv for the full per-agent
echo presence trace across every stage, if any agents are missing.
pause
