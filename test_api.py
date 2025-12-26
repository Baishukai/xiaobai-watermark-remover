"""
API测试脚本
用于测试去水印接口的功能和性能
"""
import requests
import time
import sys
from pathlib import Path

API_URL = "http://localhost:8000/api/remove-watermark"

def test_single_image(image_path: str, output_path: str = "test_result.jpg"):
    """测试单张图片处理"""
    print(f"正在处理图片: {image_path}")
    
    if not Path(image_path).exists():
        print(f"错误: 图片文件不存在 - {image_path}")
        return False
    
    start_time = time.time()
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            data = {'watermark_ratio': 0.15}
            response = requests.post(API_URL, files=files, data=data, timeout=60)
        
        if response.status_code == 200:
            with open(output_path, 'wb') as out:
                out.write(response.content)
            elapsed = time.time() - start_time
            print(f"✓ 处理成功！耗时: {elapsed:.2f}秒")
            print(f"  结果已保存到: {output_path}")
            return True
        else:
            print(f"✗ 处理失败: {response.status_code}")
            try:
                error = response.json()
                print(f"  错误信息: {error}")
            except:
                print(f"  响应内容: {response.text[:200]}")
            return False
    
    except requests.exceptions.RequestException as e:
        print(f"✗ 请求失败: {e}")
        return False
    except Exception as e:
        print(f"✗ 发生错误: {e}")
        return False

def test_health():
    """测试健康检查接口"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✓ 服务运行正常")
            print(f"  响应: {response.json()}")
            return True
        else:
            print(f"✗ 健康检查失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ 无法连接到服务: {e}")
        print("  请确保服务已启动: python main.py")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("图片去水印API测试工具")
    print("=" * 50)
    print()
    
    # 测试健康检查
    print("1. 测试服务健康状态...")
    if not test_health():
        sys.exit(1)
    print()
    
    # 测试图片处理
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else "test_result.jpg"
    else:
        # 使用默认测试图片
        test_image = Path("右下角有水印.png")
        if test_image.exists():
            image_path = str(test_image)
            output_path = "test_result.png"
        else:
            print("错误: 请提供测试图片路径")
            print("用法: python test_api.py <图片路径> [输出路径]")
            sys.exit(1)
    
    print(f"2. 测试图片处理...")
    test_single_image(image_path, output_path)
    print()
    print("测试完成！")

