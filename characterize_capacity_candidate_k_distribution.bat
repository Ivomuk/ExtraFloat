@echo off
python scripts\characterize_capacity_candidate_k_distribution.py %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: characterize_capacity_candidate_k_distribution.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

pause
