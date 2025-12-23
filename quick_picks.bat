@echo off
title Kalshi Quick Picks
cd /d "%~dp0"
echo Fetching top 10 Kalshi picks...
echo.
python -m kalshi_crawler.runner --picks 10
echo.
pause
