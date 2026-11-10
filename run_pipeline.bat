@echo off
title HengSeng Pipeline CLI Runner
cls
set "PYTHON_EXE=C:\Users\sysum\AppData\Local\Programs\Python\Python313\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"
set "PATH=C:\Users\sysum\AppData\Local\Programs\Python\Python313;C:\Users\sysum\AppData\Local\Programs\Python\Python313\Scripts;%PATH%"

"%PYTHON_EXE%" run_pipeline.py
pause
