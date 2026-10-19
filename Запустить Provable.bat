@echo off
rem Двойной щелчок — запускает Provable и открывает его в браузере.
cd /d "%~dp0"
python serve.py --open
pause
