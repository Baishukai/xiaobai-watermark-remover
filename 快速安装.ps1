# 快速安装脚本 - 自动设置执行策略并运行
# 使用方法：右键点击此文件 -> 使用 PowerShell 运行

# 临时允许脚本执行
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "图片去水印工具 - 快速安装" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 检查 Python 是否安装
try {
    $pythonVersion = python --version 2>&1
    Write-Host "检测到 Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "错误: 未检测到 Python，请先安装 Python 3.7+!" -ForegroundColor Red
    Read-Host "按 Enter 键退出"
    exit 1
}

Write-Host "正在创建虚拟环境..." -ForegroundColor Yellow
python -m venv venv

if (-not (Test-Path "venv\Scripts\Activate.ps1")) {
    Write-Host "错误: 虚拟环境创建失败!" -ForegroundColor Red
    Read-Host "按 Enter 键退出"
    exit 1
}

Write-Host "正在激活虚拟环境..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1

Write-Host "正在升级 pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip -q

Write-Host "正在安装依赖包..." -ForegroundColor Yellow
pip install -r requirements.txt

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "安装完成！" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "启动服务方法：" -ForegroundColor Cyan
Write-Host "  1. 运行: .\快速启动.ps1" -ForegroundColor Yellow
Write-Host "  2. 或运行: python main.py" -ForegroundColor Yellow
Write-Host ""
Read-Host "按 Enter 键退出"

