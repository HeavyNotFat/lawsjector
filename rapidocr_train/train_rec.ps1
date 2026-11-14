param(
    [string]$Config = "configs\rec_finetune.yml"
)

$Root = $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Repo = Join-Path $Root "third_party\PaddleOCR"
$ConfigPath = Join-Path $Root $Config

if (-not (Test-Path $Python)) { Write-Error "找不到训练环境：$Python，请先运行 setup.ps1"; exit 1 }
if (-not (Test-Path $ConfigPath)) { Write-Error "找不到配置文件：$ConfigPath"; exit 1 }

Write-Host "训练 rec：$ConfigPath" -ForegroundColor Cyan
Write-Host "日志：$Root\logs\rec_train.log"

Set-Location $Repo
$env:FLAGS_allocator_strategy = "auto_growth"
& $Python "tools/train.py" -c $ConfigPath 2>&1 | Tee-Object -FilePath (Join-Path $Root "logs\rec_train.log") -Append
exit $LASTEXITCODE
