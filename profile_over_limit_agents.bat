@echo off
python scripts\profile_over_limit_agents.py %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: profile_over_limit_agents.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

pause
