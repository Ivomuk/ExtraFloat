@echo off
python scripts\compare_persona_severity_diagnostic.py %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: compare_persona_severity_diagnostic.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

pause
