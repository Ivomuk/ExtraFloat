@echo off
python scripts\derive_capacity_function_and_backtest_frontier.py %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: derive_capacity_function_and_backtest_frontier.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

pause
