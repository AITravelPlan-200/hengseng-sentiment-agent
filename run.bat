@echo off
title HengSeng Sentiment Price Prediction Agent
cls
echo =======================================================================
echo   HengSeng Sentiment Price Prediction Agent
echo   Starting Institutional Financial Dashboard...
echo =======================================================================

set "PYTHON_EXE=C:\Users\sysum\AppData\Local\Programs\Python\Python313\python.exe"
if not exist "%PYTHON_EXE%" (
    where python >nul 2>nul
    if %ERRORLEVEL% equ 0 (
        set "PYTHON_EXE=python"
    ) else (
        echo [ERROR] Python executable not found.
        pause
        exit /b 1
    )
)

set "PATH=C:\Users\sysum\AppData\Local\Programs\Python\Python313;C:\Users\sysum\AppData\Local\Programs\Python\Python313\Scripts;%PATH%"

echo.
echo [1/2] Checking database and baseline state...
"%PYTHON_EXE%" -c "from database.db import db; print('Database ready. Records loaded.')"

echo.
echo [2/2] Launching Streamlit web interface...
echo Local Terminal URL: http://localhost:8501
echo.
"%PYTHON_EXE%" -m streamlit run app.py --server.port 8501

pause
