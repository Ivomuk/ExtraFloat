@echo off
python -m extrafloat.cap_proposal.run_burden_trial %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: run_burden_trial failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

pause
