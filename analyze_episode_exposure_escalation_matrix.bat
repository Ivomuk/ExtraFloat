@echo off
python scripts\analyze_episode_exposure_escalation_matrix.py %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: analyze_episode_exposure_escalation_matrix.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

pause
