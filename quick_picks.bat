@echo off
title Kalshi Quick Picks
cd /d "C:\Users\sonso\isoengine"
echo ============================================
echo     KALSHI QUICK PICKS - Crawl + Analyze
echo ============================================
echo.
echo Pulling latest code...
git pull origin claude/kalshi-web-crawler-Cl45i
echo.
echo Crawling all sources (this takes ~1 min)...
python -m kalshi_crawler.runner --once
echo.
echo ============================================
echo TOP 10 PICKS
echo ============================================
python -m kalshi_crawler.runner --picks 10
echo.
pause
