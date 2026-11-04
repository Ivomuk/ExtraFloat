@echo off
python -m extrafloat.cap_proposal.summarize_vs_disbursed %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: summarize_vs_disbursed failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

pause
