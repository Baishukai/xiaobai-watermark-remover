# 安装依赖（不需要 opencv-python）
# 这个方案使用 Pillow 来实现去水印功能

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "安装依赖（Pillow 方案，无需 opencv）" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

python --version
Write-Host ""

Write-Host "升级 pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

Write-Host "安装基础依赖..." -ForegroundColor Yellow
pip install fastapi==0.104.1
pip install "uvicorn[standard]==0.24.0"
pip install python-multipart==0.0.6

Write-Host "安装 Pillow..." -ForegroundColor Yellow
pip install "Pillow>=10.2.0" --prefer-binary

Write-Host "安装 numpy..." -ForegroundColor Yellow
pip install "numpy>=1.24.0" --prefer-binary

Write-Host ""
Write-Host "验证安装..." -ForegroundColor Yellow
python -c "import fastapi, uvicorn, PIL, numpy; print('All packages imported successfully')"

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "安装完成！" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "注意：此方案使用 Pillow 实现去水印，无需 opencv-python" -ForegroundColor Yellow
Write-Host "效果略逊于 OpenCV 方案，但可以正常使用" -ForegroundColor Yellow
Write-Host ""
Write-Host "启动服务: python main.py" -ForegroundColor Cyan

