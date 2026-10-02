@echo off
if "%1"=="" (
    echo Usage: analyze_persona_k8_forward_outcomes.bat ^<forward_outcomes_csv^>
    echo   e.g.: analyze_persona_k8_forward_outcomes.bat data\persona_k8_forward_outcomes.csv
    echo   ^(export data\persona_k8_forward_outcomes_query.sql's result to that path first^)
    exit /b 1
)

python scripts\analyze_persona_k8_forward_outcomes.py --forward-outcomes-file %1

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: analyze_persona_k8_forward_outcomes.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo Complete. See segmentation_outputs\persona_k8_profile\persona_k8_forward_outcomes_summary.csv
pause
