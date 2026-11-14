param(
    [string]$VsDevCmd = ""
)

$Root = Split-Path -Parent $PSScriptRoot
$Ort = Join-Path $PSScriptRoot "third_party\onnxruntime-win-x64-1.20.1"
$Source = Join-Path $PSScriptRoot "lawsjector_native.cpp"
$OutDir = Join-Path $Root "resources\native"
$Dll = Join-Path $OutDir "lawsjector_native.dll"

if (-not $VsDevCmd) {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $vsPath = & $vswhere -latest -products * -property installationPath
        if ($vsPath) { $VsDevCmd = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat" }
    }
}
if (-not (Test-Path $VsDevCmd)) {
    Write-Error "找不到 Visual Studio 的 vcvars64.bat，请安装 VS Build Tools 或用 -VsDevCmd 指定路径"
    exit 1
}
if (-not (Test-Path (Join-Path $Ort "include\onnxruntime_cxx_api.h"))) {
    Write-Error "缺少 ONNX Runtime SDK：$Ort，下载方式见 native\README.md"
    exit 1
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$compile = 'cl /nologo /O2 /MT /EHsc /std:c++17 /utf-8 /LD "' + $Source + '" ' +
    '/I"' + (Join-Path $Ort "include") + '" ' +
    '/Fo:"' + (Join-Path $OutDir "lawsjector_native.obj") + '" ' +
    '/Fe:"' + $Dll + '" ' +
    '/link "' + (Join-Path $Ort "lib\onnxruntime.lib") + '" /INCREMENTAL:NO'
$out = cmd /c ('"' + $VsDevCmd + '" >nul && ' + $compile) 2>&1
$out | Out-String | Write-Host

if (-not (Test-Path $Dll)) {
    Write-Error "构建失败"
    exit 1
}
Copy-Item (Join-Path $Ort "lib\onnxruntime.dll") (Join-Path $OutDir "onnxruntime.dll") -Force
Remove-Item (Join-Path $OutDir "lawsjector_native.obj"), `
            (Join-Path $OutDir "lawsjector_native.lib"), `
            (Join-Path $OutDir "lawsjector_native.exp") -Force -ErrorAction SilentlyContinue
Write-Host ("构建成功：{0}" -f $Dll) -ForegroundColor Green
