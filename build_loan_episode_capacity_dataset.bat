@echo off
python scripts\build_loan_episode_capacity_dataset.py %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: build_loan_episode_capacity_dataset.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

pause
