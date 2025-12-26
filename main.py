"""
FastAPI后端主程序
支持图片去水印API接口
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from PIL import Image, ImageFilter, ImageEnhance
import io
import base64
import re
import numpy as np
from typing import Optional
import uvicorn
from pathlib import Path
import os
import socket

# 设置环境变量，强制使用 CPU（必须在导入 torch 之前）
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['FORCE_CUDA'] = '0'

# 尝试导入 opencv，如果没有则使用 Pillow 方案
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

# 尝试导入 LaMa 模型
HAS_LAMA = False
_lama_model = None
_torch_device = None
_local_model_path = None

try:
    import torch
    from simple_lama_inpainting.utils import download_model, prepare_img_and_mask
    from simple_lama_inpainting.models.model import LAMA_MODEL_URL
    
    # 强制使用 CPU
    torch.cuda.is_available = lambda: False
    _torch_device = torch.device('cpu')
    
    # 检查本地模型文件（优先使用本地模型，方便打包部署）
    _script_dir = Path(__file__).parent
    _local_model_path = _script_dir / "models" / "big-lama.pt"
    
    if _local_model_path.exists():
        print(f"[OK] Found local LaMa model: {_local_model_path}")
    else:
        _local_model_path = None
        print("[INFO] Local model not found, will download from network")
    
    HAS_LAMA = True
except ImportError as e:
    print(f"[INFO] LaMa not available: {e}")
    HAS_LAMA = False

app = FastAPI(title="图片去水印工具", version="1.0.0")

# 配置CORS，允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 启动时显示使用的方案
if HAS_LAMA:
    print("[OK] LaMa AI 模型可用（效果最佳，类似 ezremove.ai）")
elif HAS_OPENCV:
    print("[OK] 使用 OpenCV 方案")
else:
    print("[INFO] 使用 Pillow 方案（效果略逊但可用）")

# 创建上传和输出目录
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


def detect_and_create_watermark_mask(img_bgr: np.ndarray, watermark_size: int = 50) -> np.ndarray:
    """
    智能检测并创建水印 mask
    
    水印特征：
    - 四角星形状
    - 浅灰色/半透明白色
    - 位于图片最右下角
    """
    height, width = img_bgr.shape[:2]
    
    # 提取右下角区域进行分析
    margin = 5
    search_size = watermark_size + 20
    
    x1 = max(0, width - search_size)
    y1 = max(0, height - search_size)
    
    corner = img_bgr[y1:height, x1:width].copy()
    corner_gray = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
    corner_h, corner_w = corner_gray.shape
    
    # 多种方法检测水印
    
    # 1. 局部亮度差异（水印比背景略亮）
    blur = cv2.GaussianBlur(corner_gray, (21, 21), 0)
    diff = cv2.subtract(corner_gray, blur)
    _, bright_mask = cv2.threshold(diff, 3, 255, cv2.THRESH_BINARY)
    
    # 2. 边缘检测
    edges = cv2.Canny(corner_gray, 30, 100)
    
    # 3. 高亮区域（相对于局部均值）
    mean_val = np.mean(corner_gray)
    _, high_bright = cv2.threshold(corner_gray, int(mean_val + 15), 255, cv2.THRESH_BINARY)
    
    # 组合检测结果
    combined = cv2.bitwise_or(bright_mask, edges)
    
    # 形态学操作
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined = cv2.dilate(combined, kernel, iterations=1)
    
    # 查找轮廓，选择合适大小的
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    local_mask = np.zeros((corner_h, corner_w), dtype=np.uint8)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        # 水印面积约 200-2500 像素
        if 100 < area < 3000:
            x, y, w, h = cv2.boundingRect(contour)
            # 水印在右下角区域
            cx, cy = x + w//2, y + h//2
            if cx > corner_w * 0.4 and cy > corner_h * 0.4:
                cv2.drawContours(local_mask, [contour], -1, 255, -1)
    
    # 如果检测结果不足，使用固定区域
    if cv2.countNonZero(local_mask) < 100:
        # 直接标记右下角 45x45 区域
        wm_size = 45
        local_mask[corner_h - wm_size - margin:corner_h - margin, 
                   corner_w - wm_size - margin:corner_w - margin] = 255
    
    # 扩展 mask 确保完全覆盖
    local_mask = cv2.dilate(local_mask, kernel, iterations=2)
    
    # 创建全图 mask
    full_mask = np.zeros((height, width), dtype=np.uint8)
    full_mask[y1:height, x1:width] = local_mask
    
    return full_mask


def remove_watermark_with_lama(image: Image.Image, watermark_size: int = 50) -> Image.Image:
    """
    使用 LaMa AI 模型去除水印（效果最佳，类似 ezremove.ai）
    
    LaMa (Large Mask Inpainting) 是目前最先进的图像修复模型
    """
    global _lama_model, _torch_device
    
    # 延迟加载模型
    if _lama_model is None:
        print("[INFO] Loading LaMa AI model (first time may take a moment)...")
        try:
            # 优先使用本地模型，否则从网络下载
            if _local_model_path and _local_model_path.exists():
                model_path = str(_local_model_path)
                print(f"[INFO] Using local model: {model_path}")
            else:
                model_path = download_model(LAMA_MODEL_URL)
                print(f"[INFO] Model downloaded to: {model_path}")
            
            # 使用 map_location='cpu' 加载模型，避免 CUDA 问题
            model = torch.jit.load(model_path, map_location='cpu')
            model.eval()
            model.to(_torch_device)
            _lama_model = model
            print("[OK] LaMa model loaded successfully")
        except Exception as e:
            print(f"[ERROR] Failed to load LaMa model: {e}")
            # 回退到 OpenCV 方案
            return remove_watermark_with_opencv(image, watermark_size)
    
    width, height = image.size
    
    # 水印区域参数（与 OpenCV 方案保持一致）
    wm_width = 96
    wm_height = 105
    margin_right = 6
    margin_bottom = 2
    
    x2 = width - margin_right
    y2 = height - margin_bottom
    x1 = x2 - wm_width
    y1 = y2 - wm_height
    
    # 确保不越界
    x1 = max(0, x1)
    y1 = max(0, y1)
    
    # 创建水印 mask
    mask = Image.new("L", (width, height), 0)
    mask_array = np.array(mask)
    mask_array[y1:y2, x1:x2] = 255
    mask = Image.fromarray(mask_array)
    
    print(f"[INFO] LaMa processing watermark region: ({x1},{y1}) to ({x2},{y2})")
    
    try:
        # 使用官方的预处理函数
        img_tensor, mask_tensor = prepare_img_and_mask(image, mask, _torch_device)
        
        # 运行模型
        with torch.inference_mode():
            inpainted = _lama_model(img_tensor, mask_tensor)
            
            cur_res = inpainted[0].permute(1, 2, 0).detach().cpu().numpy()
            cur_res = np.clip(cur_res * 255, 0, 255).astype(np.uint8)
            
            return Image.fromarray(cur_res)
    except Exception as e:
        print(f"[ERROR] LaMa inference failed: {e}")
        # 回退到 OpenCV 方案
        return remove_watermark_with_opencv(image, watermark_size)


def texture_sample_fill(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    使用周围纹理智能填充水印区域
    
    策略：分析水印上方的纹理特征，按行复制以保持木地板纹理的连续性
    """
    result = img_bgr.copy()
    height, width = img_bgr.shape[:2]
    
    # 找到 mask 的边界
    coords = np.where(mask > 0)
    if len(coords[0]) == 0:
        return result
    
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()
    
    fill_h = y_max - y_min + 1
    fill_w = x_max - x_min + 1
    
    # 确保有足够的上方区域用于采样
    sample_height = min(fill_h + 20, y_min)  # 采样区域的高度
    if sample_height < 10:
        return result
    
    # 从正上方区域采样（保持相同的 x 坐标，这样木纹能对齐）
    sample_region = img_bgr[y_min - sample_height:y_min, x_min:x_max + 1].copy()
    sample_h, sample_w = sample_region.shape[:2]
    
    if sample_h == 0 or sample_w == 0:
        return result
    
    # 逐行填充水印区域
    for y in range(y_min, min(y_max + 1, height)):
        offset_y = y - y_min
        # 从采样区域镜像获取对应行
        sample_y = sample_h - 1 - (offset_y % sample_h)
        
        for x in range(x_min, min(x_max + 1, width)):
            if mask[y, x] > 0:
                sample_x = x - x_min
                if 0 <= sample_y < sample_h and 0 <= sample_x < sample_w:
                    result[y, x] = sample_region[sample_y, sample_x]
    
    return result


def remove_watermark_with_opencv(image: Image.Image, watermark_size: int = 50) -> Image.Image:
    """
    使用纹理采样 + OpenCV inpainting 去除右下角水印
    
    策略：
    1. 精确定位水印区域
    2. 从上方采样相同纹理填充
    3. 使用 inpainting 平滑边缘
    """
    img_array = np.array(image)
    height, width = img_array.shape[:2]
    
    # 转换为 BGR
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # 水印区域参数（根据分析结果精确调整）
    # 分析结果：水印边界框 (924,919) 到 (1018,1022)，图片 1024x1024
    # 距右边缘：1024-1018 = 6px
    # 距下边缘：1024-1022 = 2px
    # 水印大小：94x103
    wm_width = 96   # 水印宽度（稍大一点确保覆盖）
    wm_height = 105  # 水印高度
    margin_right = 6   # 距右边缘
    margin_bottom = 2  # 距下边缘
    
    # 计算水印区域边界
    x2 = width - margin_right
    y2 = height - margin_bottom
    x1 = x2 - wm_width
    y1 = y2 - wm_height
    
    # 确保不越界
    x1 = max(0, x1)
    y1 = max(0, y1)
    
    # 创建水印 mask
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    
    print(f"[INFO] Watermark region: ({x1},{y1}) to ({x2},{y2}), size: {x2-x1}x{y2-y1}")
    
    # 第一步：使用纹理采样填充
    filled = texture_sample_fill(img_bgr, mask)
    
    # 第二步：使用 inpainting 平滑整个水印区域
    # 使用较大的修复半径以获得更好的效果
    inpainted = cv2.inpaint(filled, mask, 10, cv2.INPAINT_TELEA)
    
    # 第三步：创建边缘过渡 mask 进行羽化混合
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    dilated = cv2.dilate(mask, kernel, iterations=2)
    
    # 创建羽化 mask
    feather_mask = cv2.GaussianBlur(dilated, (15, 15), 0)
    feather_mask = feather_mask.astype(np.float32) / 255.0
    
    # 混合：在水印区域使用修复结果，在边缘进行平滑过渡
    feather_mask_3ch = feather_mask[:, :, np.newaxis]
    result = (inpainted.astype(np.float32) * feather_mask_3ch + 
              img_bgr.astype(np.float32) * (1 - feather_mask_3ch))
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    # 转回 RGB
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(result_rgb)


def detect_watermark_mask_pillow(img_array: np.ndarray, search_region: tuple) -> np.ndarray:
    """
    使用 Pillow/NumPy 检测水印区域（不需要 OpenCV）
    """
    start_x, start_y, end_x, end_y = search_region
    region = img_array[start_y:end_y, start_x:end_x].copy()
    
    height, width = region.shape[:2]
    
    if len(region.shape) == 3:
        # 转换到灰度
        gray = np.mean(region, axis=2).astype(np.uint8)
        
        # 计算 HSV 近似值
        # V (亮度) 近似为 max(R,G,B)
        v = np.max(region, axis=2)
        # S (饱和度) 近似
        min_rgb = np.min(region, axis=2)
        s = np.where(v > 0, (v - min_rgb) * 255 // (v + 1), 0).astype(np.uint8)
        
        # 检测高亮度低饱和度区域
        mask = ((v > 200) & (s < 60)).astype(np.uint8) * 255
        
        # 简单的膨胀操作
        from scipy import ndimage
        mask = ndimage.binary_dilation(mask > 0, iterations=3).astype(np.uint8) * 255
    else:
        # 灰度图：检测高亮度区域
        mask = (region > 220).astype(np.uint8) * 255
        from scipy import ndimage
        mask = ndimage.binary_dilation(mask > 0, iterations=3).astype(np.uint8) * 255
    
    return mask


def inpaint_simple(img_array: np.ndarray, mask: np.ndarray, radius: int = 5) -> np.ndarray:
    """
    简单的 inpainting 实现（不需要 OpenCV）
    使用周围像素的加权平均来填充
    """
    from scipy import ndimage
    
    result = img_array.copy()
    mask_bool = mask > 0
    
    if not np.any(mask_bool):
        return result
    
    # 对每个通道分别处理
    if len(img_array.shape) == 3:
        for c in range(img_array.shape[2]):
            channel = result[:, :, c].astype(float)
            # 用周围像素的均值填充
            for _ in range(radius * 2):
                blurred = ndimage.uniform_filter(channel, size=radius)
                channel = np.where(mask_bool, blurred, channel)
            result[:, :, c] = channel.astype(np.uint8)
    else:
        channel = result.astype(float)
        for _ in range(radius * 2):
            blurred = ndimage.uniform_filter(channel, size=radius)
            channel = np.where(mask_bool, blurred, channel)
        result = channel.astype(np.uint8)
    
    return result


def remove_watermark_with_pillow(image: Image.Image, watermark_ratio: float = 0.06) -> Image.Image:
    """
    使用 Pillow 方法去除水印（不需要 opencv）
    
    改进版：自动检测水印位置，使用迭代填充算法
    """
    width, height = image.size
    
    # 搜索区域
    search_width = int(width * watermark_ratio)
    search_height = int(height * watermark_ratio)
    search_width = max(search_width, 50)
    search_height = max(search_height, 50)
    
    start_x = width - search_width
    start_y = height - search_height
    
    img_array = np.array(image)
    
    # 检测水印
    local_mask = detect_watermark_mask_pillow(img_array, (start_x, start_y, width, height))
    
    if np.sum(local_mask > 0) < 10:
        print("未检测到明显水印")
        return image
    
    # 创建全图 mask
    full_mask = np.zeros((height, width), dtype=np.uint8)
    full_mask[start_y:height, start_x:width] = local_mask
    
    # 使用简单 inpainting 修复
    result_array = inpaint_simple(img_array, full_mask, radius=7)
    
    return Image.fromarray(result_array)


def remove_watermark_with_custom_region(image: Image.Image, region: dict) -> Image.Image:
    """
    使用用户自定义的水印区域去除水印
    
    Args:
        image: PIL Image对象
        region: 水印区域坐标 {"x1": int, "y1": int, "x2": int, "y2": int}
    
    Returns:
        处理后的PIL Image对象
    """
    global _lama_model, _torch_device
    
    width, height = image.size
    
    # 获取区域坐标
    x1 = max(0, int(region.get('x1', 0)))
    y1 = max(0, int(region.get('y1', 0)))
    x2 = min(width, int(region.get('x2', width)))
    y2 = min(height, int(region.get('y2', height)))
    
    # 确保区域有效
    if x2 <= x1 or y2 <= y1:
        print(f"[WARNING] Invalid region: ({x1},{y1}) to ({x2},{y2})")
        return image
    
    print(f"[INFO] Processing custom region: ({x1},{y1}) to ({x2},{y2}), size: {x2-x1}x{y2-y1}")
    
    # 创建 mask
    mask = Image.new("L", (width, height), 0)
    mask_array = np.array(mask)
    mask_array[y1:y2, x1:x2] = 255
    mask = Image.fromarray(mask_array)
    
    # 优先使用 LaMa
    if HAS_LAMA:
        # 延迟加载模型
        if _lama_model is None:
            print("[INFO] Loading LaMa AI model...")
            try:
                if _local_model_path and _local_model_path.exists():
                    model_path = str(_local_model_path)
                else:
                    model_path = download_model(LAMA_MODEL_URL)
                
                model = torch.jit.load(model_path, map_location='cpu')
                model.eval()
                model.to(_torch_device)
                _lama_model = model
                print("[OK] LaMa model loaded successfully")
            except Exception as e:
                print(f"[ERROR] Failed to load LaMa model: {e}")
                return _remove_with_opencv_mask(image, mask)
        
        try:
            img_tensor, mask_tensor = prepare_img_and_mask(image, mask, _torch_device)
            
            with torch.inference_mode():
                inpainted = _lama_model(img_tensor, mask_tensor)
                cur_res = inpainted[0].permute(1, 2, 0).detach().cpu().numpy()
                cur_res = np.clip(cur_res * 255, 0, 255).astype(np.uint8)
                return Image.fromarray(cur_res)
        except Exception as e:
            print(f"[ERROR] LaMa inference failed: {e}")
            return _remove_with_opencv_mask(image, mask)
    
    # 回退到 OpenCV
    return _remove_with_opencv_mask(image, mask)


def remove_watermark_from_bottom_right(image: Image.Image, watermark_size: int = 50) -> Image.Image:
    """
    去除图片右下角的水印
    
    Args:
        image: PIL Image对象
        watermark_size: 水印区域大小（像素，默认 50）
    
    Returns:
        处理后的PIL Image对象
    """
    if HAS_LAMA:
        # 优先使用 LaMa AI 模型（效果最佳）
        return remove_watermark_with_lama(image, watermark_size)
    elif HAS_OPENCV:
        # 使用 OpenCV（效果中等）
        return remove_watermark_with_opencv(image, watermark_size)
    else:
        # 如果没有 OpenCV，使用 Pillow 方案
        return remove_watermark_with_pillow(image, watermark_size)


@app.get("/")
async def root():
    """根路径，返回前端页面"""
    return FileResponse("index.html")


def remove_watermark_with_custom_mask(image: Image.Image, mask: Image.Image) -> Image.Image:
    """
    使用用户自定义的 mask 去除水印
    
    Args:
        image: 原始图片
        mask: 用户绘制的 mask（白色区域表示需要去除的区域）
    
    Returns:
        处理后的图片
    """
    global _lama_model, _torch_device
    
    # 确保 mask 尺寸与图片一致
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.LANCZOS)
    
    # 转换为灰度
    if mask.mode != 'L':
        mask = mask.convert('L')
    
    # 检查 mask 是否有有效内容
    mask_array = np.array(mask)
    if np.max(mask_array) < 50:
        print("[INFO] Mask is empty, using default watermark region")
        return remove_watermark_from_bottom_right(image)
    
    print(f"[INFO] Using custom mask, white pixels: {np.sum(mask_array > 128)}")
    
    # 优先使用 LaMa
    if HAS_LAMA:
        # 延迟加载模型
        if _lama_model is None:
            print("[INFO] Loading LaMa AI model...")
            try:
                if _local_model_path and _local_model_path.exists():
                    model_path = str(_local_model_path)
                    print(f"[INFO] Using local model: {model_path}")
                else:
                    model_path = download_model(LAMA_MODEL_URL)
                    print(f"[INFO] Model downloaded to: {model_path}")
                
                model = torch.jit.load(model_path, map_location='cpu')
                model.eval()
                model.to(_torch_device)
                _lama_model = model
                print("[OK] LaMa model loaded successfully")
            except Exception as e:
                print(f"[ERROR] Failed to load LaMa model: {e}")
                # 回退到 OpenCV
                return _remove_with_opencv_mask(image, mask)
        
        try:
            # 使用 LaMa 修复
            img_tensor, mask_tensor = prepare_img_and_mask(image, mask, _torch_device)
            
            with torch.inference_mode():
                inpainted = _lama_model(img_tensor, mask_tensor)
                cur_res = inpainted[0].permute(1, 2, 0).detach().cpu().numpy()
                cur_res = np.clip(cur_res * 255, 0, 255).astype(np.uint8)
                return Image.fromarray(cur_res)
        except Exception as e:
            print(f"[ERROR] LaMa inference failed: {e}")
            return _remove_with_opencv_mask(image, mask)
    
    # 回退到 OpenCV
    return _remove_with_opencv_mask(image, mask)


def _remove_with_opencv_mask(image: Image.Image, mask: Image.Image) -> Image.Image:
    """使用 OpenCV inpainting 和自定义 mask"""
    if not HAS_OPENCV:
        print("[WARNING] OpenCV not available, returning original image")
        return image
    
    img_array = np.array(image)
    mask_array = np.array(mask.convert('L'))
    
    # 转换为 BGR
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # 二值化 mask
    _, mask_binary = cv2.threshold(mask_array, 128, 255, cv2.THRESH_BINARY)
    
    # 使用 inpainting
    inpainted = cv2.inpaint(img_bgr, mask_binary, 10, cv2.INPAINT_TELEA)
    
    # 转回 RGB
    result_rgb = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(result_rgb)


@app.post("/api/detect-watermark")
async def detect_watermark(
    file: UploadFile = File(...),
):
    """
    检测水印位置 API
    
    返回检测到的水印区域坐标，用于前端预览
    """
    try:
        # 验证文件类型
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="只支持图片文件")
        
        # 读取上传的文件
        contents = await file.read()
        
        # 打开图片
        image = Image.open(io.BytesIO(contents))
        width, height = image.size
        
        # 使用固定参数检测水印位置（与去水印算法保持一致）
        # 水印区域参数（根据分析结果精确调整）
        wm_width = 96   # 水印宽度
        wm_height = 105  # 水印高度
        margin_right = 6   # 距右边缘
        margin_bottom = 2  # 距下边缘
        
        # 计算水印区域边界
        x2 = width - margin_right
        y2 = height - margin_bottom
        x1 = x2 - wm_width
        y1 = y2 - wm_height
        
        # 确保不越界
        x1 = max(0, x1)
        y1 = max(0, y1)
        
        print(f"[INFO] Detected watermark region: ({x1},{y1}) to ({x2},{y2})")
        
        return {
            "success": True,
            "image_size": {"width": width, "height": height},
            "watermark_region": {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "width": x2 - x1,
                "height": y2 - y1
            },
            "message": "水印区域检测完成"
        }
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"检测失败: {str(e)}")


@app.post("/api/remove-watermark")
async def remove_watermark(
    file: UploadFile = File(...),
    mode: Optional[str] = Form(default="auto"),
    mask: Optional[str] = Form(default=None),
    watermark_size: Optional[int] = Form(default=50),
    custom_region: Optional[str] = Form(default=None)
):
    """
    去水印API接口
    
    Args:
        file: 上传的图片文件
        mode: 处理模式 ('auto' 或 'manual')
        mask: Base64 编码的 mask 图片（手动模式）
        watermark_size: 水印区域大小（像素，默认50，自动模式使用）
    
    Returns:
        处理后的图片文件流
    """
    try:
        # 验证文件类型
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="只支持图片文件")
        
        # 读取上传的文件
        contents = await file.read()
        
        # 打开图片
        image = Image.open(io.BytesIO(contents))
        
        # 如果是RGBA模式，转换为RGB
        if image.mode == 'RGBA':
            # 创建白色背景
            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
            rgb_image.paste(image, mask=image.split()[3])  # 使用alpha通道作为mask
            image = rgb_image
        elif image.mode not in ['RGB', 'L']:
            image = image.convert('RGB')
        
        print(f"[INFO] Processing image: {image.size}, mode: {mode}")
        
        # 根据模式处理
        if mode == 'auto' and custom_region:
            # 自动模式 + 用户自定义区域
            try:
                import json
                region = json.loads(custom_region)
                print(f"[INFO] Using custom region: {region}")
                processed_image = remove_watermark_with_custom_region(image, region)
            except Exception as e:
                print(f"[ERROR] Failed to parse custom region: {e}")
                processed_image = remove_watermark_from_bottom_right(image, watermark_size or 50)
        elif mode == 'manual' and mask:
            # 手动模式：使用用户绘制的 mask
            try:
                # 解析 Base64 mask
                # 格式: data:image/png;base64,xxxx
                if ',' in mask:
                    mask_data = mask.split(',')[1]
                else:
                    mask_data = mask
                
                mask_bytes = base64.b64decode(mask_data)
                mask_image = Image.open(io.BytesIO(mask_bytes))
                
                print(f"[INFO] Custom mask received: {mask_image.size}")
                
                # 使用自定义 mask 去水印
                processed_image = remove_watermark_with_custom_mask(image, mask_image)
            except Exception as e:
                print(f"[ERROR] Failed to parse mask: {e}")
                # 回退到自动模式
                processed_image = remove_watermark_from_bottom_right(image, watermark_size or 50)
        else:
            # 自动模式：使用默认的水印位置
            watermark_size = max(30, min(100, watermark_size or 50))
            processed_image = remove_watermark_from_bottom_right(image, watermark_size)
        
        # 将处理后的图片转换为字节流
        output = io.BytesIO()
        # 保持原始格式或转换为JPEG
        if file.content_type == 'image/png':
            processed_image.save(output, format='PNG')
            media_type = 'image/png'
        else:
            processed_image.save(output, format='JPEG', quality=95)
            media_type = 'image/jpeg'
        
        output.seek(0)
        
        # 返回图片流
        return StreamingResponse(
            output,
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename=processed_{file.filename}"
            }
        )
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "ok", "message": "服务正常运行"}


if __name__ == "__main__":
    def _is_port_available(host: str, port: int) -> bool:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((host, port))
            return True
        except OSError:
            return False

    def _pick_port(host: str, preferred_port: int) -> int:
        # 依次尝试 preferred_port, preferred_port+1, ... preferred_port+49
        for p in range(preferred_port, preferred_port + 50):
            if _is_port_available(host, p):
                return p
        raise RuntimeError(f"找不到可用端口（从 {preferred_port} 到 {preferred_port + 49} 都被占用）")

    host = os.getenv("HOST", "0.0.0.0")
    preferred_port = int(os.getenv("PORT", "9000"))
    port = _pick_port(host, preferred_port)
    if port != preferred_port:
        print(f"⚠ 端口 {preferred_port} 已被占用，自动切换到 {port}（可通过设置环境变量 PORT 指定）")

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )

