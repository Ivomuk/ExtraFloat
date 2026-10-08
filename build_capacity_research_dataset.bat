@echo off
python scripts\build_capacity_research_dataset.py %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: build_capacity_research_dataset.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

pause
