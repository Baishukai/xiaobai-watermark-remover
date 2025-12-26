"""
测试去水印算法
"""
from PIL import Image
import numpy as np
import cv2
import sys
sys.path.insert(0, '.')

# Reload module to get latest changes
import importlib
import main
importlib.reload(main)

from main import remove_watermark_from_bottom_right, detect_and_create_watermark_mask, HAS_OPENCV

print(f"OpenCV available: {HAS_OPENCV}")

# Load the test image
input_path = r"测试用的图片\原始的带有星星水印的图片.png"
output_path = "test_result_opencv.png"

print(f"Loading: {input_path}")
img = Image.open(input_path)
print(f"Size: {img.size}")

# Remove watermark
print("Processing with OpenCV...")
result = remove_watermark_from_bottom_right(img, watermark_size=55)

# Save result
result.save(output_path)
print(f"Done! Saved to: {output_path}")

# Also save the mask for debugging
img_array = np.array(img)
img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
mask = detect_and_create_watermark_mask(img_bgr, 55)
cv2.imwrite("debug_watermark_mask.png", mask)
print(f"Debug mask saved to: debug_watermark_mask.png")

