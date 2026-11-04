@echo off
python -m extrafloat.cap_proposal.explain_headroom %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: explain_headroom failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

pause
