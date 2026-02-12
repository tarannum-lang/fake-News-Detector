@echo off
echo Starting Fake News Detector...
python main.py
if %errorlevel% neq 0 (
    echo.
    echo Application exited with error code %errorlevel%
    pause
)
