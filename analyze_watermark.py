"""
分析水印区域的像素特征
"""
import cv2
import numpy as np
from PIL import Image

# 尝试加载有水印的原图
for path in ["processed-image.png", "右下角有水印.png"]:
    try:
        img = cv2.imread(path)
        if img is None:
            continue
        print(f"\n=== Analyzing: {path} ===")
        print(f"Image shape: {img.shape}")
        
        height, width = img.shape[:2]
        
        # 提取右下角 100x100 区域
        corner_size = 100
        corner = img[height-corner_size:height, width-corner_size:width]
        corner_gray = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
        
        print(f"Corner region shape: {corner.shape}")
        print(f"Corner gray min: {corner_gray.min()}, max: {corner_gray.max()}, mean: {corner_gray.mean():.1f}")
        
        # 保存角落区域用于查看
        cv2.imwrite(f"corner_{path}", corner)
        print(f"Corner saved to: corner_{path}")
        
        # 分析像素分布
        hist = cv2.calcHist([corner_gray], [0], None, [256], [0, 256])
        peak_idx = np.argmax(hist)
        print(f"Most common gray value: {peak_idx}")
        
        # 检测边缘
        edges = cv2.Canny(corner_gray, 30, 100)
        print(f"Canny edges non-zero: {cv2.countNonZero(edges)}")
        cv2.imwrite(f"edges_{path}", edges)
        
        # 检测与均值的差异
        blur = cv2.GaussianBlur(corner_gray, (21, 21), 0)
        diff = cv2.absdiff(corner_gray, blur)
        print(f"Diff from blur - min: {diff.min()}, max: {diff.max()}, mean: {diff.mean():.2f}")
        
        # 不同阈值的结果
        for thresh in [3, 5, 8, 10, 15]:
            _, mask = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)
            print(f"  Threshold {thresh}: {cv2.countNonZero(mask)} pixels")
        
    except Exception as e:
        print(f"Error with {path}: {e}")

