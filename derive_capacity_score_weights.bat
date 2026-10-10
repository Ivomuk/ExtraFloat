@echo off
python scripts\derive_capacity_score_weights.py %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: derive_capacity_score_weights.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

pause
