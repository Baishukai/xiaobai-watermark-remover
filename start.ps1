# PowerShell 启动脚本
Write-Host "正在启动图片去水印服务..." -ForegroundColor Green

# 检查虚拟环境是否存在
if (-not (Test-Path "venv\Scripts\Activate.ps1")) {
    Write-Host "错误: 虚拟环境不存在，请先运行 .\setup.ps1" -ForegroundColor Red
    Read-Host "按 Enter 键退出"
    exit 1
}

# 激活虚拟环境
& .\venv\Scripts\Activate.ps1

# 启动服务
python main.py

