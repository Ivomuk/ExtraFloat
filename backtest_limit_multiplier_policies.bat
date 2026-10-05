@echo off
python scripts\backtest_limit_multiplier_policies.py %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: backtest_limit_multiplier_policies.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

pause
