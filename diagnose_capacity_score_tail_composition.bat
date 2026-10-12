@echo off
python scripts\diagnose_capacity_score_tail_composition.py %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: diagnose_capacity_score_tail_composition.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

pause
