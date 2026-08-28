$ErrorActionPreference = 'Stop'
$UiRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$UiPort = 8501
Set-Location -LiteralPath $UiRoot

python -c "import streamlit, plotly" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Error 'Missing dependencies. Run: python -m pip install -r requirements.txt'
    exit 1
}

$PortBusy = $false
$Probe = [System.Net.Sockets.TcpClient]::new()
try {
    $Probe.Connect('127.0.0.1', $UiPort)
    $PortBusy = $true
} catch [System.Net.Sockets.SocketException] {
    $PortBusy = $false
} finally {
    $Probe.Dispose()
}
if ($PortBusy) {
    Write-Host "[ERROR] Port $UiPort is already in use. Stop the existing local process or choose another port before starting the demonstration." -ForegroundColor Red
    exit 2
}

Write-Host "Starting HDC System Demonstration at http://127.0.0.1:$UiPort"
python -m streamlit run app.py --server.address 127.0.0.1 --server.port $UiPort --server.headless true --browser.gatherUsageStats false
