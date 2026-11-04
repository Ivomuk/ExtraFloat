@echo off
python -m extrafloat.cap_proposal.validate_burden_history %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: validate_burden_history failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

pause
