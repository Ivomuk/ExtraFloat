@echo off
python -m extrafloat.cap_proposal.compare_scenarios %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: compare_scenarios failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

pause
