# 一键安装依赖（Windows / PowerShell）
# 用法：右键"使用 PowerShell 运行"，或在项目根目录执行：
#     powershell -ExecutionPolicy Bypass -File .\install.ps1
$ErrorActionPreference = "Stop"

Write-Host "[1/4] 创建虚拟环境 .venv ..." -ForegroundColor Cyan
if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    python -m venv .venv
}

$py = ".\.venv\Scripts\python.exe"

Write-Host "[2/4] 升级 pip ..." -ForegroundColor Cyan
& $py -m pip install --upgrade pip

Write-Host "[3/4] 安装 torch / torchvision（CPU 版，体积更小）..." -ForegroundColor Cyan
& $py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

Write-Host "[4/4] 安装其余依赖 ..." -ForegroundColor Cyan
& $py -m pip install -r requirements.txt

Write-Host "完成！在 VSCode 中选择解释器： .\.venv\Scripts\python.exe" -ForegroundColor Green
