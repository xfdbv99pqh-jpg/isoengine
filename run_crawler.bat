@echo off
title Kalshi Crawler
cd /d "%~dp0"

echo ============================================
echo         KALSHI CRAWLER LAUNCHER
echo ============================================
echo.
echo Choose an option:
echo   1. Run once (crawl all sources)
echo   2. Show top 10 picks
echo   3. Show top 5 picks
echo   4. Interactive shell
echo   5. Generate strategy report
echo   6. Run scheduled (continuous)
echo   7. Exit
echo.

set /p choice="Enter choice (1-7): "

if "%choice%"=="1" (
    python -m kalshi_crawler.runner --once
    pause
    goto :eof
)
if "%choice%"=="2" (
    python -m kalshi_crawler.runner --picks 10
    pause
    goto :eof
)
if "%choice%"=="3" (
    python -m kalshi_crawler.runner --picks 5
    pause
    goto :eof
)
if "%choice%"=="4" (
    python -m kalshi_crawler.runner --shell
    goto :eof
)
if "%choice%"=="5" (
    python -m kalshi_crawler.runner --strategy
    pause
    goto :eof
)
if "%choice%"=="6" (
    echo Starting continuous crawl... Press Ctrl+C to stop
    python -m kalshi_crawler.runner
    goto :eof
)
if "%choice%"=="7" (
    exit
)

echo Invalid choice
pause
