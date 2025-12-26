# PowerShell 安装脚本
Write-Host "正在创建虚拟环境..." -ForegroundColor Green
python -m venv venv

Write-Host "正在激活虚拟环境..." -ForegroundColor Green
& .\venv\Scripts\Activate.ps1

Write-Host "正在升级pip..." -ForegroundColor Green
python -m pip install --upgrade pip

Write-Host "正在安装依赖包..." -ForegroundColor Green
pip install -r requirements.txt

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "安装完成！" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "运行 .\start.ps1 启动服务" -ForegroundColor Yellow
Write-Host "或者运行: python main.py" -ForegroundColor Yellow
Write-Host ""
Read-Host "按 Enter 键退出"

