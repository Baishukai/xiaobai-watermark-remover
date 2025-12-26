"""
对比原图和理想效果图，分析水印位置
"""
import cv2
import numpy as np

# 加载图片（使用 PIL 处理中文路径）
from PIL import Image
import os

folder = os.path.join(os.path.dirname(__file__), "测试用的图片")
orig_path = os.path.join(folder, "原始的带有星星水印的图片.png")
ideal_path = os.path.join(folder, "理想的处理后的图片.png")

orig_pil = Image.open(orig_path)
ideal_pil = Image.open(ideal_path)

original = cv2.cvtColor(np.array(orig_pil), cv2.COLOR_RGB2BGR)
ideal = cv2.cvtColor(np.array(ideal_pil), cv2.COLOR_RGB2BGR)

if original is None or ideal is None:
    print("Failed to load images")
    exit()

print(f"Original shape: {original.shape}")
print(f"Ideal shape: {ideal.shape}")

# 计算差异
diff = cv2.absdiff(original, ideal)
diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

# 找到差异最大的区域
_, thresh = cv2.threshold(diff_gray, 5, 255, cv2.THRESH_BINARY)

# 找到边界
coords = np.where(thresh > 0)
if len(coords[0]) > 0:
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()
    print(f"Watermark region: y=[{y_min}, {y_max}], x=[{x_min}, {x_max}]")
    print(f"Watermark size: {y_max - y_min + 1} x {x_max - x_min + 1}")
    print(f"Distance from bottom: {original.shape[0] - y_max - 1}")
    print(f"Distance from right: {original.shape[1] - x_max - 1}")
else:
    print("No difference found")

# 保存差异图
cv2.imwrite("diff_image.png", diff * 10)  # 放大差异以便观察
cv2.imwrite("diff_thresh.png", thresh)
print("Saved diff_image.png and diff_thresh.png")

# 提取右下角 100x100 区域
h, w = original.shape[:2]
corner_orig = original[h-100:h, w-100:w]
corner_ideal = ideal[h-100:h, w-100:w]
cv2.imwrite("corner_original.png", corner_orig)
cv2.imwrite("corner_ideal.png", corner_ideal)
print("Saved corner images")

