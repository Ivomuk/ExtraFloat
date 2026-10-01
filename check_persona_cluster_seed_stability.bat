@echo off
python scripts\check_persona_cluster_seed_stability.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: check_persona_cluster_seed_stability.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

pause
