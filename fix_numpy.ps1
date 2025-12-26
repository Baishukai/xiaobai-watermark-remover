# 修复 numpy 和 opencv-python 兼容性问题
# 降级 numpy 到 1.x 版本以兼容 opencv-python 4.8.1.78

Write-Host "正在卸载当前的 numpy..." -ForegroundColor Yellow
pip uninstall numpy -y

Write-Host "正在安装兼容的 numpy 1.x 版本..." -ForegroundColor Yellow
# 尝试安装最新的 numpy 1.x 版本，优先使用预编译包
pip install "numpy>=1.24.0,<2.0.0" --prefer-binary --only-binary :all:

if ($LASTEXITCODE -ne 0) {
    Write-Host "预编译包安装失败，尝试安装特定版本..." -ForegroundColor Yellow
    # 如果预编译包不可用，尝试安装特定版本
    pip install numpy==1.26.4 --prefer-binary
}

Write-Host "验证安装..." -ForegroundColor Yellow
python -c "import numpy; print(f'✓ numpy {numpy.__version__} 安装成功')"
python -c "import cv2; print(f'✓ opencv-python {cv2.__version__} 导入成功')"

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "修复完成！" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

