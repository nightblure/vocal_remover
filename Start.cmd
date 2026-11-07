@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Install Python 3.11 x64, uv and FFmpeg with ffprobe and libmp3lame.
    echo Then run in this folder: uv sync --frozen
    echo See docs\launch.md for Windows requirements and installation instructions.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" "src\stem_local\launcher.py" --root "%CD%" %*
set "RESULT=%ERRORLEVEL%"
if not "%RESULT%"=="0" (
    echo.
    echo Application failed. See docs\launch.md and the messages above.
    pause
)
exit /b %RESULT%
