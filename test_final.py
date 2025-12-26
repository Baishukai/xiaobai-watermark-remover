"""
测试最终的去水印算法
"""
from PIL import Image
import numpy as np
import cv2
import sys
from pathlib import Path

# 获取脚本目录
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

# 重新加载模块
import importlib
import main
importlib.reload(main)

from main import remove_watermark_with_opencv, HAS_OPENCV

print(f"OpenCV available: {HAS_OPENCV}")

# 使用中文路径
test_dir = script_dir / "测试用的图片"
input_path = test_dir / "原始的带有星星水印的图片.png"
ideal_path = test_dir / "理想的处理后的图片.png"
output_path = script_dir / "test_final_result.png"

print(f"Loading original: {input_path}")
img = Image.open(input_path)
print(f"Size: {img.size}")

# Remove watermark
print("Processing...")
result = remove_watermark_with_opencv(img)

# Save result
result.save(output_path)
print(f"Done! Saved to: {output_path}")

# 对比结果
print("\nComparing with ideal result...")
ideal = Image.open(ideal_path)
result_array = np.array(result)
ideal_array = np.array(ideal)

# 计算差异
diff = np.abs(result_array.astype(np.int16) - ideal_array.astype(np.int16))
mean_diff = np.mean(diff)
max_diff = np.max(diff)
print(f"Mean difference from ideal: {mean_diff:.2f}")
print(f"Max difference from ideal: {max_diff}")

# 保存差异可视化
diff_vis = np.clip(diff * 5, 0, 255).astype(np.uint8)
Image.fromarray(diff_vis).save(script_dir / "diff_from_ideal.png")
print(f"Difference visualization saved to: diff_from_ideal.png")

