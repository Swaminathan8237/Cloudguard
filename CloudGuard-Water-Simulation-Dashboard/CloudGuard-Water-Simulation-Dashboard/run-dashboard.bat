@echo off
setlocal
cd /d "%~dp0"
echo CloudGuard-Water Simulation Dashboard
 echo.
echo Serving at http://127.0.0.1:5173
 echo Keep this terminal open during the presentation.
echo.
python -m http.server 5173 --bind 127.0.0.1
pause
