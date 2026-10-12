@echo off
python scripts\combine_capacity_candidate_k_with_frozen_c3.py %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: combine_capacity_candidate_k_with_frozen_c3.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

pause
