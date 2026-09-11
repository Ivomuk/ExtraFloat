@echo off
echo === Checking model output against the August whitelist/blacklist ===

python scripts\check_whitelist_blacklist_eval.py --whitelist-file data\whitelist_aug_20260804.csv --blacklist-file data\blacklist_aug_20260804.csv --output-file output\engine_test_output.csv --borrower-file borrower_history_retail_filtered.csv --score-col cal_pd

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: check_whitelist_blacklist_eval.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo Whitelist/blacklist check complete. See wl_bl_eval_matched_agents.csv and the JSON baseline summary above.
pause
