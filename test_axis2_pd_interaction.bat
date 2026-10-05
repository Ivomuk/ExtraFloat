@echo off
python scripts\test_axis2_pd_interaction.py %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: test_axis2_pd_interaction.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

pause
