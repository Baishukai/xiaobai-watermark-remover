# 快速启动脚本 - 自动设置执行策略并运行
# 使用方法：右键点击此文件 -> 使用 PowerShell 运行

# 临时允许脚本执行
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "图片去水印工具 - 启动服务" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 检查虚拟环境是否存在
if (-not (Test-Path "venv\Scripts\Activate.ps1")) {
    Write-Host "错误: 虚拟环境不存在!" -ForegroundColor Red
    Write-Host "请先运行 快速安装.ps1 进行安装" -ForegroundColor Yellow
    Read-Host "按 Enter 键退出"
    exit 1
}

Write-Host "正在激活虚拟环境..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1

Write-Host "正在启动服务..." -ForegroundColor Yellow
Write-Host "服务地址: http://localhost:8000" -ForegroundColor Green
Write-Host "按 Ctrl+C 停止服务" -ForegroundColor Cyan
Write-Host ""

# 启动服务
python main.py

