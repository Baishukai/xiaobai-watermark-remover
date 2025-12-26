"""
测试 LaMa 模型修复效果
"""
import os
import sys

# 设置环境变量，强制使用 CPU（必须在导入 torch 之前）
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['FORCE_CUDA'] = '0'

# 设置控制台输出编码
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except:
        pass

import numpy as np
from PIL import Image

def test_lama():
    """测试 LaMa 模型"""
    print("=" * 60)
    print("LaMa Image Inpainting Model Test")
    print("=" * 60)
    
    # 1. 尝试导入 torch 并强制 CPU
    print("\n[1/5] Setting up PyTorch for CPU...")
    try:
        import torch
        # 禁用 CUDA
        torch.cuda.is_available = lambda: False
        device = torch.device('cpu')
        print(f"[OK] PyTorch version: {torch.__version__}, Device: {device}")
    except Exception as e:
        print(f"[FAIL] PyTorch setup failed: {e}")
        return False
    
    # 2. 尝试导入 simple_lama_inpainting 并修复 CPU 加载问题
    print("\n[2/5] Importing simple_lama_inpainting...")
    try:
        from simple_lama_inpainting.utils import download_model
        from simple_lama_inpainting.models.model import LAMA_MODEL_URL
        from simple_lama_inpainting.utils import prepare_img_and_mask
        print("[OK] Import successful")
    except ImportError as e:
        print(f"[FAIL] Import failed: {e}")
        print("Please run: pip install simple-lama-inpainting")
        return False
    
    # 3. 加载模型（优先使用本地模型）
    print("\n[3/5] Loading LaMa model...")
    try:
        # 优先使用本地模型（使用 pathlib 处理路径，避免编码问题）
        from pathlib import Path
        script_dir = Path(__file__).resolve().parent
        local_model_path = script_dir / "models" / "big-lama.pt"
        
        if local_model_path.exists():
            model_path = str(local_model_path)
            print(f"  Using local model: {model_path}")
        else:
            model_path = download_model(LAMA_MODEL_URL)
            print(f"  Model downloaded to: {model_path}")
        
        # 使用 map_location='cpu' 加载模型，避免 CUDA 问题
        model = torch.jit.load(model_path, map_location='cpu')
        model.eval()
        model.to(device)
        
        # 创建一个包装类来使用加载的模型
        class SimpleLamaCPU:
            def __init__(self, model, device):
                self.model = model
                self.device = device
            
            def __call__(self, image, mask):
                # 使用官方的预处理函数
                img_tensor, mask_tensor = prepare_img_and_mask(image, mask, self.device)
                
                # 运行模型
                with torch.inference_mode():
                    inpainted = self.model(img_tensor, mask_tensor)
                    
                    cur_res = inpainted[0].permute(1, 2, 0).detach().cpu().numpy()
                    cur_res = np.clip(cur_res * 255, 0, 255).astype(np.uint8)
                    
                    return Image.fromarray(cur_res)
        
        lama = SimpleLamaCPU(model, device)
        print("[OK] Model loaded successfully (CPU mode)")
    except Exception as e:
        print(f"[FAIL] Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 4. 加载测试图片
    print("\n[4/5] Loading test image...")
    test_image_path = os.path.join(os.path.dirname(__file__), "测试用的图片", "原始的带有星星水印的图片.png")
    if not os.path.exists(test_image_path):
        print(f"[FAIL] Test image not found: {test_image_path}")
        return False
    
    image = Image.open(test_image_path).convert("RGB")
    width, height = image.size
    print(f"[OK] Image loaded: {width}x{height}")
    
    # 5. 创建水印 mask 并测试修复
    print("\n[5/5] Testing LaMa inpainting...")
    
    # 水印区域参数（与 main.py 保持一致）
    wm_width = 96
    wm_height = 105
    margin_right = 6
    margin_bottom = 2
    
    x2 = width - margin_right
    y2 = height - margin_bottom
    x1 = x2 - wm_width
    y1 = y2 - wm_height
    
    # 创建 mask（白色区域表示需要修复的区域）
    mask = Image.new("L", (width, height), 0)
    mask_array = np.array(mask)
    mask_array[y1:y2, x1:x2] = 255
    mask = Image.fromarray(mask_array)
    
    print(f"  Watermark region: ({x1},{y1}) to ({x2},{y2})")
    
    try:
        # 使用 LaMa 修复
        result = lama(image, mask)
        
        # 保存结果
        output_path = os.path.join(os.path.dirname(__file__), "test_lama_result.png")
        result.save(output_path)
        print(f"[OK] LaMa inpainting successful! Result saved to: {output_path}")
        
        # 与理想效果对比
        ideal_path = os.path.join(os.path.dirname(__file__), "测试用的图片", "理想的处理后的图片.png")
        if os.path.exists(ideal_path):
            ideal = Image.open(ideal_path).convert("RGB")
            
            # 计算水印区域的差异
            result_array = np.array(result)
            ideal_array = np.array(ideal)
            
            # 只比较水印区域
            region_result = result_array[y1:y2, x1:x2]
            region_ideal = ideal_array[y1:y2, x1:x2]
            
            diff = np.abs(region_result.astype(float) - region_ideal.astype(float))
            avg_diff = np.mean(diff)
            
            print(f"\nAverage difference from ideal (watermark region): {avg_diff:.2f}")
            if avg_diff < 10:
                print("[EXCELLENT] Almost identical to ideal result!")
            elif avg_diff < 20:
                print("[GOOD] Close to ideal result")
            else:
                print("[OK] May need parameter adjustment")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] LaMa inpainting failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_lama()
    print("\n" + "=" * 60)
    if success:
        print("Test completed! LaMa model is working properly")
    else:
        print("Test failed, please check error messages")
    print("=" * 60)

