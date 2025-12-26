# 专门用于 Python 3.14 的依赖安装脚本
# 解决 numpy 等包的编译问题

Write-Host "正在升级 pip 和构建工具..." -ForegroundColor Yellow
python -m pip install --upgrade pip setuptools wheel

Write-Host "正在安装 numpy（使用预编译包）..." -ForegroundColor Yellow
# 先单独安装 numpy，强制使用预编译包
pip install numpy --only-binary :all: --prefer-binary

Write-Host "正在安装其他依赖..." -ForegroundColor Yellow
# 然后安装其他依赖
pip install fastapi==0.104.1
pip install "uvicorn[standard]==0.24.0"
pip install python-multipart==0.0.6
pip install "Pillow>=10.2.0"
pip install opencv-python==4.8.1.78

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "依赖安装完成！" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

