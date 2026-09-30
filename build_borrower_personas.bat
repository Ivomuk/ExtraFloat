@echo off
python segmentation\borrower_persona_clustering.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: borrower_persona_clustering.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo Complete. Output written to segmentation_outputs\borrower_persona_output\
pause
