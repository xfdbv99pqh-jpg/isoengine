@echo off
title Kalshi Crawler
cd /d "C:\Users\sonso\isoengine"

echo ============================================
echo         KALSHI CRAWLER LAUNCHER
echo ============================================
echo.
echo Pulling latest code...
git pull origin claude/kalshi-web-crawler-Cl45i
echo.
echo ============================================
echo.
echo Choose an option:
echo   1. Crawl + Show top 10 picks (RECOMMENDED)
echo   2. Crawl + Show top 5 picks
echo   3. Crawl only (no picks)
echo   4. Show picks only (use cached data)
echo   5. Interactive shell
echo   6. Generate strategy report
echo   7. Run scheduled (continuous)
echo   8. Exit
echo.

set /p choice="Enter choice (1-8): "

if "%choice%"=="1" (
    echo Crawling all sources...
    python -m kalshi_crawler.runner --once
    echo.
    echo ============================================
    echo Generating picks...
    echo ============================================
    python -m kalshi_crawler.runner --picks 10
    pause
    goto :eof
)
if "%choice%"=="2" (
    echo Crawling all sources...
    python -m kalshi_crawler.runner --once
    echo.
    echo ============================================
    echo Generating picks...
    echo ============================================
    python -m kalshi_crawler.runner --picks 5
    pause
    goto :eof
)
if "%choice%"=="3" (
    python -m kalshi_crawler.runner --once
    pause
    goto :eof
)
if "%choice%"=="4" (
    python -m kalshi_crawler.runner --picks 10
    pause
    goto :eof
)
if "%choice%"=="5" (
    python -m kalshi_crawler.runner --shell
    goto :eof
)
if "%choice%"=="6" (
    python -m kalshi_crawler.runner --once
    python -m kalshi_crawler.runner --strategy
    pause
    goto :eof
)
if "%choice%"=="7" (
    echo Starting continuous crawl... Press Ctrl+C to stop
    python -m kalshi_crawler.runner
    goto :eof
)
if "%choice%"=="8" (
    exit
)

echo Invalid choice
pause
