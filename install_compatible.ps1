# 兼容性安装脚本 - 针对 Python 3.14
# 先安装 numpy 2.4.0（已确认有预编译包），然后安装 opencv-python（忽略依赖）

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "安装兼容版本（Python 3.14）" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 升级 pip
Write-Host "升级 pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# 先卸载 opencv-python
Write-Host "卸载旧版本的 opencv-python..." -ForegroundColor Yellow
pip uninstall opencv-python opencv-python-headless -y 2>$null

# 确保安装 numpy 2.4.0（已确认有预编译包）
Write-Host "安装 numpy 2.4.0（预编译版本）..." -ForegroundColor Yellow
pip install numpy==2.4.0 --upgrade --force-reinstall --prefer-binary

# 尝试安装 opencv-python，忽略依赖检查（使用 --no-deps 然后手动安装）
Write-Host "尝试安装 opencv-python（忽略依赖检查）..." -ForegroundColor Yellow
pip install opencv-python --no-deps --prefer-binary

# 如果上面成功，验证是否兼容
Write-Host ""
Write-Host "验证安装..." -ForegroundColor Yellow
python -c "import numpy; print(f'✓ numpy 版本: {numpy.__version__}')" 2>$null
if ($LASTEXITCODE -eq 0) {
    python -c "import cv2; print(f'✓ opencv-python 版本: {cv2.__version__}')" 2>$null
    if ($LASTEXITCODE -eq 0) {
        python -c "import cv2, numpy as np; img = np.zeros((10,10), dtype=np.uint8); print('✓ 兼容性测试成功')" 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host ""
            Write-Host "========================================" -ForegroundColor Green
            Write-Host "安装成功！" -ForegroundColor Green
            Write-Host "========================================" -ForegroundColor Green
            exit 0
        }
    }
}

# 如果上面的方法失败，尝试其他方案
Write-Host "方案1失败，尝试使用 opencv-python-headless..." -ForegroundColor Yellow
pip uninstall opencv-python -y 2>$null
pip install opencv-python-headless --no-deps --prefer-binary

# 再次验证
python -c "import cv2; print(f'✓ opencv-python-headless 版本: {cv2.__version__}')" 2>$null
if ($LASTEXITCODE -eq 0) {
    python -c "import cv2, numpy as np; img = np.zeros((10,10), dtype=np.uint8); print('✓ 兼容性测试成功')" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Green
        Write-Host "使用 opencv-python-headless 安装成功！" -ForegroundColor Green
        Write-Host "========================================" -ForegroundColor Green
        exit 0
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Red
Write-Host "安装失败！" -ForegroundColor Red
Write-Host "========================================" -ForegroundColor Red
Write-Host ""
Write-Host "Python 3.14 太新，opencv-python 的依赖要求与 numpy 2.4.0 不兼容。" -ForegroundColor Yellow
Write-Host "建议：" -ForegroundColor Yellow
Write-Host "1. 使用 Python 3.11 或 3.12（兼容性更好）" -ForegroundColor Yellow
Write-Host "2. 或者等待相关包的更新" -ForegroundColor Yellow

