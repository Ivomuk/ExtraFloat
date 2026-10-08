@echo off
python scripts\check_ceiling_binding_by_category.py %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: check_ceiling_binding_by_category.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

pause
