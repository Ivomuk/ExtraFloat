@echo off
python scripts\check_enhanced_limits_vs_over_limit.py %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: check_enhanced_limits_vs_over_limit.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

pause
