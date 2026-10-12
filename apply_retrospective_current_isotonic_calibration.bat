@echo off
python scripts\apply_retrospective_current_isotonic_calibration.py %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: apply_retrospective_current_isotonic_calibration.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

pause
