@echo off
python scripts\check_loan_episode_dataset_integrity.py %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: check_loan_episode_dataset_integrity.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

pause
