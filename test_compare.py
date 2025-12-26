"""
对比测试：OpenCV vs LaMa 去水印效果
"""
import os
import sys

# 设置环境变量
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# 设置控制台编码
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except:
        pass

import numpy as np
from PIL import Image

def main():
    print("=" * 70)
    print("Watermark Removal Comparison: OpenCV vs LaMa")
    print("=" * 70)
    
    # 加载测试图片
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_image_path = os.path.join(script_dir, "测试用的图片", "原始的带有星星水印的图片.png")
    ideal_path = os.path.join(script_dir, "测试用的图片", "理想的处理后的图片.png")
    
    if not os.path.exists(test_image_path):
        print(f"[ERROR] Test image not found: {test_image_path}")
        return
    
    original = Image.open(test_image_path).convert("RGB")
    print(f"[OK] Original image loaded: {original.size}")
    
    # 加载理想效果（用于对比）
    ideal = None
    if os.path.exists(ideal_path):
        ideal = Image.open(ideal_path).convert("RGB")
        print(f"[OK] Ideal result loaded for comparison")
    
    # 水印区域参数
    width, height = original.size
    wm_width = 96
    wm_height = 105
    margin_right = 6
    margin_bottom = 2
    x1 = width - margin_right - wm_width
    y1 = height - margin_bottom - wm_height
    x2 = width - margin_right
    y2 = height - margin_bottom
    
    print(f"\nWatermark region: ({x1},{y1}) to ({x2},{y2})")
    
    # 导入去水印模块
    print("\n" + "-" * 70)
    print("Testing methods...")
    print("-" * 70)
    
    results = {}
    
    # 1. 测试 OpenCV 方案
    print("\n[1] Testing OpenCV method...")
    try:
        from main import remove_watermark_with_opencv
        result_opencv = remove_watermark_with_opencv(original.copy())
        output_path = os.path.join(script_dir, "compare_opencv_result.png")
        result_opencv.save(output_path)
        results['OpenCV'] = result_opencv
        print(f"    [OK] Saved to: {output_path}")
    except Exception as e:
        print(f"    [FAIL] {e}")
    
    # 2. 测试 LaMa 方案
    print("\n[2] Testing LaMa method...")
    try:
        from main import remove_watermark_with_lama, HAS_LAMA
        if HAS_LAMA:
            result_lama = remove_watermark_with_lama(original.copy())
            output_path = os.path.join(script_dir, "compare_lama_result.png")
            result_lama.save(output_path)
            results['LaMa'] = result_lama
            print(f"    [OK] Saved to: {output_path}")
        else:
            print("    [SKIP] LaMa not available")
    except Exception as e:
        print(f"    [FAIL] {e}")
        import traceback
        traceback.print_exc()
    
    # 计算与理想效果的差异
    if ideal is not None:
        print("\n" + "-" * 70)
        print("Comparison with ideal result (lower is better)")
        print("-" * 70)
        
        ideal_array = np.array(ideal)
        
        for name, result in results.items():
            result_array = np.array(result)
            
            # 只比较水印区域
            region_result = result_array[y1:y2, x1:x2]
            region_ideal = ideal_array[y1:y2, x1:x2]
            
            diff = np.abs(region_result.astype(float) - region_ideal.astype(float))
            avg_diff = np.mean(diff)
            max_diff = np.max(diff)
            
            # 评级
            if avg_diff < 10:
                rating = "EXCELLENT"
            elif avg_diff < 20:
                rating = "GOOD"
            elif avg_diff < 30:
                rating = "OK"
            else:
                rating = "NEEDS IMPROVEMENT"
            
            print(f"\n  {name}:")
            print(f"    Average difference: {avg_diff:.2f}")
            print(f"    Max difference: {max_diff:.2f}")
            print(f"    Rating: [{rating}]")
    
    print("\n" + "=" * 70)
    print("Comparison complete! Check the output images:")
    for name in results:
        print(f"  - compare_{name.lower()}_result.png")
    print("=" * 70)


if __name__ == "__main__":
    main()

