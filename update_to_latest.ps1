# 更新到最新的兼容版本（Python 3.14）
# 确保使用预编译的 numpy 和最新兼容的 opencv-python

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "更新到最新兼容版本" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 升级 pip 和构建工具
Write-Host "升级 pip 和构建工具..." -ForegroundColor Yellow
python -m pip install --upgrade pip setuptools wheel

# 检查当前 numpy 版本
Write-Host "检查当前 numpy 版本..." -ForegroundColor Yellow
try {
    python -c "import numpy; print(f'当前 numpy 版本: {numpy.__version__}')"
} catch {
    Write-Host "numpy 未安装"
}

# 卸载旧版本的 opencv-python
Write-Host "卸载旧版本的 opencv-python..." -ForegroundColor Yellow
pip uninstall opencv-python opencv-python-headless -y 2>$null

# 策略：先尝试安装最新版本的 opencv-python，让它自动处理 numpy 依赖
Write-Host "尝试安装最新版本的 opencv-python（让它自动处理 numpy 依赖）..." -ForegroundColor Yellow
pip install --upgrade opencv-python --prefer-binary

if ($LASTEXITCODE -ne 0) {
    Write-Host "方案1失败，尝试手动安装兼容版本..." -ForegroundColor Yellow
    
    # 方案2：尝试安装 numpy 2.2.x（如果 opencv-python 需要）
    Write-Host "安装 numpy 2.2.x（预编译版本）..." -ForegroundColor Yellow
    pip uninstall numpy -y 2>$null
    
    # 尝试多个可能的版本
    $numpyVersions = @("2.2.6", "2.2.5", "2.2.4", "2.2.3")
    $installed = $false
    
    foreach ($version in $numpyVersions) {
        Write-Host "尝试安装 numpy $version..." -ForegroundColor Yellow
        pip install "numpy==$version" --prefer-binary --only-binary :all: 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ 成功安装 numpy $version" -ForegroundColor Green
            $installed = $true
            break
        }
    }
    
    if (-not $installed) {
        Write-Host "所有预编译版本都失败，尝试使用最新的 numpy 2.x..." -ForegroundColor Yellow
        pip install "numpy>=2.0.0,<2.3.0" --prefer-binary
    }
    
    # 再次尝试安装 opencv-python
    Write-Host "安装 opencv-python..." -ForegroundColor Yellow
    pip install --upgrade opencv-python --prefer-binary
}

Write-Host ""
Write-Host "验证安装..." -ForegroundColor Yellow
python -c "import numpy; print(f'✓ numpy 版本: {numpy.__version__}')"
python -c "import cv2; print(f'✓ opencv-python 版本: {cv2.__version__}')"
python -c "import cv2, numpy as np; img = np.zeros((10,10), dtype=np.uint8); print('✓ 兼容性测试成功')"

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "更新完成！" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "如果仍有问题，可能需要：" -ForegroundColor Yellow
Write-Host "1. 检查 opencv-python 的最新版本要求" -ForegroundColor Yellow
Write-Host "2. 或者考虑使用 Python 3.11/3.12 以获得更好的兼容性" -ForegroundColor Yellow

