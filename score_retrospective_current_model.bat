@echo off
python scripts\score_retrospective_current_model.py %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: score_retrospective_current_model.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

pause
