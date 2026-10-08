@echo off
python scripts\check_supported_exposure_boundary_robustness.py %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: check_supported_exposure_boundary_robustness.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

pause
