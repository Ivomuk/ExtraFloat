@echo off
python scripts\check_mart_column_completeness.py %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: check_mart_column_completeness.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

pause
