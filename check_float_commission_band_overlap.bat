@echo off
python scripts\check_float_commission_band_overlap.py %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: check_float_commission_band_overlap.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

pause
