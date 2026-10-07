@echo off
python scripts\export_champion_vs_challenger_limits.py %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: export_champion_vs_challenger_limits.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

pause
