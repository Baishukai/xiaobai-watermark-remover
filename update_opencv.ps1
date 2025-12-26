# 更新 opencv-python 到最新版本，兼容 numpy 2.x
# 专门为 Python 3.14 设计

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "更新 opencv-python 以兼容 numpy 2.4.0" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 检查当前 numpy 版本
Write-Host "检查当前 numpy 版本..." -ForegroundColor Yellow
python -c "import numpy; print(f'当前 numpy 版本: {numpy.__version__}')"

# 卸载旧版本的 opencv-python
Write-Host "卸载旧版本的 opencv-python..." -ForegroundColor Yellow
pip uninstall opencv-python opencv-python-headless -y

# 升级到最新版本的 opencv-python（支持 numpy 2.x）
Write-Host "安装最新版本的 opencv-python..." -ForegroundColor Yellow
pip install --upgrade opencv-python --no-deps

# 如果上面失败了，尝试安装特定版本
if ($LASTEXITCODE -ne 0) {
    Write-Host "尝试安装 opencv-python 4.10.0+..." -ForegroundColor Yellow
    pip install "opencv-python>=4.10.0" --no-deps --force-reinstall
}

Write-Host ""
Write-Host "验证安装..." -ForegroundColor Yellow
python -c "import numpy; print(f'✓ numpy 版本: {numpy.__version__}')"
python -c "import cv2; print(f'✓ opencv-python 版本: {cv2.__version__}')"
python -c "import cv2, numpy; print('✓ 导入测试成功，兼容性正常')"

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "更新完成！" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

