"""
分析原图和理想效果的像素差异，精确定位水印
"""
import cv2
import numpy as np
from pathlib import Path
import os

# 获取脚本所在目录
script_dir = Path(__file__).parent

# 使用 pathlib 处理中文路径
test_dir = script_dir / "测试用的图片"
original_path = str(test_dir / "原始的带有星星水印的图片.png")
ideal_path = str(test_dir / "理想的处理后的图片.png")

print(f"Looking for files in: {test_dir}")
print(f"Original path exists: {Path(original_path).exists()}")
print(f"Ideal path exists: {Path(ideal_path).exists()}")

# 读取两张图片
original = cv2.imdecode(np.fromfile(original_path, dtype=np.uint8), cv2.IMREAD_COLOR)
ideal = cv2.imdecode(np.fromfile(ideal_path, dtype=np.uint8), cv2.IMREAD_COLOR)

print(f"Original shape: {original.shape}")
print(f"Ideal shape: {ideal.shape}")

# 计算差异
diff = cv2.absdiff(original, ideal)
diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

# 找到有差异的区域
_, thresh = cv2.threshold(diff_gray, 5, 255, cv2.THRESH_BINARY)

# 保存差异图
cv2.imwrite("diff_visualization.png", diff * 10)  # 放大差异便于查看
cv2.imwrite("diff_mask.png", thresh)

# 找到差异区域的边界
coords = cv2.findNonZero(thresh)
if coords is not None:
    x, y, w, h = cv2.boundingRect(coords)
    print(f"\nWatermark bounding box:")
    print(f"  Top-left: ({x}, {y})")
    print(f"  Size: {w} x {h}")
    print(f"  Bottom-right: ({x+w}, {y+h})")
    
    height, width = original.shape[:2]
    print(f"\nRelative to image edges:")
    print(f"  Distance from right edge: {width - (x + w)} px")
    print(f"  Distance from bottom edge: {height - (y + h)} px")
    print(f"  Distance from right edge to watermark start: {width - x} px")
    print(f"  Distance from bottom edge to watermark start: {height - y} px")
    
    # 提取水印区域
    watermark_region = original[y:y+h, x:x+w]
    cv2.imwrite("watermark_extracted.png", watermark_region)
    
    # 分析水印像素值
    wm_gray = cv2.cvtColor(watermark_region, cv2.COLOR_BGR2GRAY)
    print(f"\nWatermark pixel analysis:")
    print(f"  Min: {wm_gray.min()}, Max: {wm_gray.max()}, Mean: {wm_gray.mean():.1f}")
    
    # 分析差异区域的像素数量
    diff_pixels = cv2.countNonZero(thresh)
    print(f"\nTotal different pixels: {diff_pixels}")
    
    # 画出水印区域的轮廓
    result = original.copy()
    cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imwrite("watermark_location.png", result)
    print(f"\nSaved: watermark_location.png")
else:
    print("No difference found!")

