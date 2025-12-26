# 快速修复：安装 numpy 2.4.0 然后尝试安装 opencv-python（忽略依赖）

Write-Host "安装 numpy 2.4.0..." -ForegroundColor Yellow
pip install numpy==2.4.0 --upgrade --force-reinstall --prefer-binary

Write-Host "卸载旧版 opencv-python..." -ForegroundColor Yellow
pip uninstall opencv-python opencv-python-headless -y 2>$null

Write-Host "尝试安装 opencv-python（忽略依赖）..." -ForegroundColor Yellow
pip install opencv-python --no-deps --prefer-binary

Write-Host "测试兼容性..." -ForegroundColor Yellow
python -c "import numpy; import cv2; print(f'numpy: {numpy.__version__}'); print(f'opencv: {cv2.__version__}'); print('✓ 导入成功')"

