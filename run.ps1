# PowerShell One-Click Launcher for HengSeng Sentiment Price Prediction Agent

$pythonPath = "C:\Users\sysum\AppData\Local\Programs\Python\Python313\python.exe"
if (-not (Test-Path $pythonPath)) {
    $pythonPath = "python"
}

$env:Path = "C:\Users\sysum\AppData\Local\Programs\Python\Python313;C:\Users\sysum\AppData\Local\Programs\Python\Python313\Scripts;" + $env:Path

Write-Host "=======================================================================" -ForegroundColor Cyan
Write-Host "  HengSeng Sentiment Price Prediction Agent - Financial Terminal       " -ForegroundColor Cyan
Write-Host "=======================================================================" -ForegroundColor Cyan
Write-Host "Local URL: http://localhost:8501" -ForegroundColor Green
Write-Host ""

& $pythonPath -m streamlit run app.py --server.port 8501
