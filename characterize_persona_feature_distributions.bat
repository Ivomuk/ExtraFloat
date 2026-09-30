@echo off
python scripts\characterize_persona_feature_distributions.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: characterize_persona_feature_distributions.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

pause
