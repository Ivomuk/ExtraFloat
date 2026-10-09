@echo off
python scripts\derive_capacity_frontier_from_business_state.py %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: derive_capacity_frontier_from_business_state.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

pause
