@echo off
python scripts\profile_persona_k8.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: profile_persona_k8.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo Complete. See segmentation_outputs\persona_k8_profile\ for the 4 artifacts.
pause
