# 修复兼容性问题：使用旧版 opencv-python（支持 numpy 1.x）+ numpy 1.x
# 这是 Python 3.14 的唯一可行方案

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "修复兼容性问题" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "卸载当前的 numpy 和 opencv-python..." -ForegroundColor Yellow
pip uninstall numpy opencv-python opencv-python-headless -y 2>$null

Write-Host "安装 opencv-python 4.8.1.78（支持 numpy 1.x）..." -ForegroundColor Yellow
pip install opencv-python==4.8.1.78 --prefer-binary

Write-Host "尝试安装 numpy 1.x（如果有预编译包）..." -ForegroundColor Yellow
# 尝试最新的 numpy 1.x 版本
pip install "numpy>=1.24.0,<2.0.0" --prefer-binary --only-binary :all:

if ($LASTEXITCODE -ne 0) {
    Write-Host "预编译包不可用，尝试安装特定版本..." -ForegroundColor Yellow
    # 尝试几个可能的版本
    $versions = @("1.26.4", "1.26.3", "1.26.2", "1.25.2")
    $success = $false
    foreach ($v in $versions) {
        Write-Host "尝试 numpy $v..." -ForegroundColor Yellow
        pip install "numpy==$v" --prefer-binary --only-binary :all: 2>$null
        if ($LASTEXITCODE -eq 0) {
            $success = $true
            break
        }
    }
    
    if (-not $success) {
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Red
        Write-Host "错误：Python 3.14 没有 numpy 1.x 的预编译包！" -ForegroundColor Red
        Write-Host "========================================" -ForegroundColor Red
        Write-Host ""
        Write-Host "解决方案：" -ForegroundColor Yellow
        Write-Host "1. 使用 Python 3.11 或 3.12（推荐）" -ForegroundColor Yellow
        Write-Host "2. 或者等待相关包的更新" -ForegroundColor Yellow
        Write-Host "3. 或者安装 Visual Studio Build Tools 来编译 numpy" -ForegroundColor Yellow
        exit 1
    }
}

Write-Host ""
Write-Host "验证安装..." -ForegroundColor Yellow
python -c "import numpy; print(f'✓ numpy 版本: {numpy.__version__}')"
python -c "import cv2; print(f'✓ opencv-python 版本: {cv2.__version__}')"
python -c "import cv2, numpy as np; img = np.zeros((10,10), dtype=np.uint8); print('✓ 兼容性测试成功')"

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "安装成功！" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

