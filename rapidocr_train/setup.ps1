$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$Venv = Join-Path $Root ".venv"
$Python = Join-Path $Venv "Scripts\python.exe"
$Repo = Join-Path $Root "third_party\PaddleOCR"
$env:UV_DEFAULT_INDEX = "https://pypi.tuna.tsinghua.edu.cn/simple"

function Step($name) { Write-Host "`n$name" -ForegroundColor Cyan }

Step "创建 Python 3.10 虚拟环境"
if (-not (Test-Path $Python)) {
    if (Get-Command uv -ErrorAction SilentlyContinue) {
        uv venv --python 3.10 $Venv
    } else {
        $py = (Get-Command py -ErrorAction SilentlyContinue)
        if (-not $py) { Write-Error "需要安装 Python 3.10（https://www.python.org/downloads/release/python-31011/）"; exit 1 }
        py -3.10 -m venv $Venv
    }
}

Step "安装 PaddlePaddle-GPU 2.6.2"
$paddleOk = $false
try {
    & $Python -c "import paddle; assert paddle.is_compiled_with_cuda(); print('paddle ok', paddle.__version__)" | Out-Host
    $paddleOk = $true
} catch { $paddleOk = $false }
if (-not $paddleOk) {
    if (Get-Command uv -ErrorAction SilentlyContinue) {
        uv pip install --python $Python "paddlepaddle-gpu==2.6.2" "numpy==1.26.4" "protobuf==3.20.2" setuptools wheel
    } else {
        & $Python -m pip install -i $env:UV_DEFAULT_INDEX "paddlepaddle-gpu==2.6.2" "numpy==1.26.4" "protobuf==3.20.2" setuptools wheel
    }
}

Step "克隆 PaddleOCR v2.10.0"
if (-not (Test-Path (Join-Path $Repo "tools\train.py"))) {
    git clone --depth 1 --branch v2.10.0 https://github.com/PaddlePaddle/PaddleOCR.git $Repo
}

Step "安装训练依赖与 ONNX 导出工具"
$deps = @(
    "-r", (Join-Path $Repo "requirements.txt"),
    "numpy==1.26.4",
    "albumentations==1.3.1",   # 1.4+ 在 2026 版 albucore 上导入报错
    "paddle2onnx==1.3.1",
    "onnx==1.16.2",
    "onnxruntime==1.20.1",
    "onnxslim",
    "rapidocr_onnxruntime==1.4.4"
)
if (Get-Command uv -ErrorAction SilentlyContinue) {
    uv pip install --python $Python @deps
} else {
    & $Python -m pip install -i $env:UV_DEFAULT_INDEX @deps
}

Step "下载 PP-OCRv4 预训练权重"
& $Python (Join-Path $Root "scripts\download_pretrained.py")

Step "生成合成数据与训练配置"
if (-not (Test-Path (Join-Path $Root "data\corpus\法治理论知识题库.txt"))) {
    Copy-Item (Join-Path $Root "..\resources\法治理论知识题库.txt") (Join-Path $Root "data\corpus\") -ErrorAction SilentlyContinue
}
& $Python (Join-Path $Root "scripts\build_rec_dataset.py") --preview
& $Python (Join-Path $Root "scripts\build_det_dataset.py") --preview
& $Python (Join-Path $Root "scripts\make_configs.py")

Write-Host "`n部署完成" -ForegroundColor Green
Write-Host "训练识别模型：powershell -File scripts\train_rec.ps1"
Write-Host "训练检测模型：powershell -File scripts\train_det.ps1"
Write-Host "导出 ONNX：.venv\Scripts\python.exe scripts\export_onnx.py --kind both"
Write-Host "对比评估：.venv\Scripts\python.exe scripts\eval_onnx.py"
