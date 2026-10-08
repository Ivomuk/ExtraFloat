@echo off
python scripts\fit_capacity_challenger_model.py %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: fit_capacity_challenger_model.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

pause
