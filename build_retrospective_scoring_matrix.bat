@echo off
python scripts\build_retrospective_scoring_matrix.py %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: build_retrospective_scoring_matrix.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

pause
