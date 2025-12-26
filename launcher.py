"""
小白AI图片去水印 - 启动器
双击运行即可启动服务并打开浏览器
"""
import os
import sys
import socket
import threading
import webbrowser
import time
import io

# 修复 PyInstaller --windowed 模式下 stdin/stdout 为 None 的问题
if sys.stdin is None:
    sys.stdin = io.StringIO()
if sys.stdout is None:
    sys.stdout = open(os.devnull, 'w')
if sys.stderr is None:
    sys.stderr = open(os.devnull, 'w')

# 设置环境变量（必须在导入其他模块之前）
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['FORCE_CUDA'] = '0'

# 获取程序运行目录
if getattr(sys, 'frozen', False):
    # 打包后的 exe 运行
    BASE_DIR = os.path.dirname(sys.executable)
else:
    # 脚本直接运行
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 切换工作目录
os.chdir(BASE_DIR)


def is_port_available(host: str, port: int) -> bool:
    """检查端口是否可用"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
        return True
    except OSError:
        return False


def find_available_port(host: str, start_port: int) -> int:
    """查找可用端口"""
    for port in range(start_port, start_port + 50):
        if is_port_available(host, port):
            return port
    raise RuntimeError(f"找不到可用端口（{start_port}-{start_port + 49} 都被占用）")


def open_browser(url: str, delay: float = 2.0):
    """延迟打开浏览器"""
    time.sleep(delay)
    print(f"\n🌐 正在打开浏览器: {url}")
    webbrowser.open(url)


def main():
    print("=" * 50)
    print("✨ 小白AI图片去水印 - 启动中...")
    print("=" * 50)
    print(f"📁 工作目录: {BASE_DIR}")
    
    # 检查必要文件
    if not os.path.exists("index.html"):
        print("❌ 错误: 找不到 index.html 文件")
        print("请确保程序在正确的目录下运行")
        input("按回车键退出...")
        sys.exit(1)
    
    # 查找可用端口
    host = "127.0.0.1"
    port = find_available_port(host, 9000)
    url = f"http://{host}:{port}"
    
    print(f"🚀 启动服务: {url}")
    print("-" * 50)
    print("💡 提示: 关闭此窗口将停止服务")
    print("-" * 50)
    
    # 延迟打开浏览器
    browser_thread = threading.Thread(target=open_browser, args=(url,))
    browser_thread.daemon = True
    browser_thread.start()
    
    # 启动服务（导入并运行 uvicorn）
    try:
        import uvicorn
        from main import app
        
        # 自定义日志配置，避免 isatty 问题
        log_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(levelname)s - %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
            },
            "loggers": {
                "uvicorn": {"handlers": ["default"], "level": "INFO"},
                "uvicorn.error": {"level": "INFO"},
                "uvicorn.access": {"handlers": ["default"], "level": "INFO"},
            },
        }
        
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="info",
            access_log=False,
            log_config=log_config
        )
    except KeyboardInterrupt:
        print("\n👋 服务已停止")
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        try:
            input("按回车键退出...")
        except:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()

