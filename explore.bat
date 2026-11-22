@echo off
rem Double-click to open the brain model viewer, where you can click any area to start a disorder there.
cd /d "%~dp0"
".venv\Scripts\python.exe" scripts\explore.py
pause
