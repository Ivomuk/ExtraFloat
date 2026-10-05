@echo off
python scripts\analyze_axis2_severity_signal.py %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: analyze_axis2_severity_signal.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

pause
