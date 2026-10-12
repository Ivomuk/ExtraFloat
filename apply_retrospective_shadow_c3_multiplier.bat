@echo off
python scripts\apply_retrospective_shadow_c3_multiplier.py %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: apply_retrospective_shadow_c3_multiplier.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

pause
