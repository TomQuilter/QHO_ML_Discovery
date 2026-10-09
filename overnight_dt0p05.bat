@echo off
cd /d "%~dp0"
echo Starting the overnight thimble scan. Close this window to stop it.
echo Run this file again later to continue from the last finished time.
echo.
"C:\Users\tomqu\Documents\MLandPython\envs\helios_env\python.exe" -u "%~dp0overnight_dt0p05.py"
echo.
echo The scan has stopped. If it finished, the tables are in runs\overnight_dt0p05_T0to4_N16
pause
