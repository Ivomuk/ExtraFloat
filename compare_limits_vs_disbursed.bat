@echo off
python -m extrafloat.cap_proposal.compare_limits_vs_disbursed %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: compare_limits_vs_disbursed failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

pause
