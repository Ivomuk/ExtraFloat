@echo off
echo === Comparing characteristics: WITH vs WITHOUT unresolved loan at snapshot ===
echo (within the "Defaulter not paid back in the last 30 days" cohort, from the
echo  with-features run's wl_bl_eval_matched_agents.csv)

python scripts\compare_defaulter_subgroup_characteristics.py --matched-file wl_bl_eval_matched_agents.csv

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: compare_defaulter_subgroup_characteristics.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

pause
