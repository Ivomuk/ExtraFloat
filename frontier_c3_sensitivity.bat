@echo off
python scripts\frontier_c3_sensitivity.py %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: frontier_c3_sensitivity.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

pause
