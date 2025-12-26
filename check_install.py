"""检查依赖包是否安装成功"""
try:
    import fastapi
    print(f"✓ fastapi 版本: {fastapi.__version__}")
except ImportError as e:
    print(f"✗ fastapi 导入失败: {e}")

try:
    import uvicorn
    print(f"✓ uvicorn 版本: {uvicorn.__version__}")
except ImportError as e:
    print(f"✗ uvicorn 导入失败: {e}")

try:
    import numpy
    print(f"✓ numpy 版本: {numpy.__version__}")
except ImportError as e:
    print(f"✗ numpy 导入失败: {e}")

try:
    import cv2
    print(f"✓ opencv-python 版本: {cv2.__version__}")
except ImportError as e:
    print(f"✗ opencv-python 导入失败: {e}")

try:
    from PIL import Image
    import PIL
    print(f"✓ Pillow 版本: {PIL.__version__}")
except ImportError as e:
    print(f"✗ Pillow 导入失败: {e}")

print("\n所有依赖包检查完成！")

