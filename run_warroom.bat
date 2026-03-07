@echo off
echo Starting War-Room Bot...
cd /d "%~dp0"
title War-Room Day Trader
if exist venv\Scripts\activate.bat call venv\Scripts\activate.bat
python main.py
pause
