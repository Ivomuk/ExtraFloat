@echo off
python scripts\derive_capacity_risk_supported_exposure_surface.py %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: derive_capacity_risk_supported_exposure_surface.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

pause
