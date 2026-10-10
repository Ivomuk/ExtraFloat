@echo off
python scripts\build_historical_pit_transaction_features.py %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: build_historical_pit_transaction_features.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

pause
