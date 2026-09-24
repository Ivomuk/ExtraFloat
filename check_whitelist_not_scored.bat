@echo off
echo === Checking whitelist agents missing from the final scored output ===
echo Direct literal check -- complementary to check_blacklist.bat's PART A,
echo which checks coverage against borrower_history.csv (an engine INPUT).
echo This checks the actual final scored file instead.

python scripts\check_whitelist_not_scored.py --whitelist-file data\whitelist_aug_20260804.csv --output-file output\engine_test_output.csv --out-missing whitelist_agents_not_scored.csv

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: check_whitelist_not_scored.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo Complete. See whitelist_agents_not_scored.csv for the full list, if any.
pause
