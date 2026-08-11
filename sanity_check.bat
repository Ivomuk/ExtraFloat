@echo off
python sanity_check.py --output output/engine_test_output.csv

if %ERRORLEVEL% neq 0 (
    echo.
    echo One or more sanity checks FAILED. Review output above.
    pause
    exit /b %ERRORLEVEL%
)

pause
