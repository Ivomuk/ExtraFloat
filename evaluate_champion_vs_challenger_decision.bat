@echo off
python scripts\evaluate_champion_vs_challenger_decision.py %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: evaluate_champion_vs_challenger_decision.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

pause
