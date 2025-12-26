"""检查 numpy 1.x 版本在 Python 3.14 上是否有预编译包"""
import subprocess
import sys

def check_numpy_version(version):
    """检查特定版本的 numpy 是否有预编译包"""
    print(f"\n检查 numpy {version}...")
    try:
        # 使用 pip 的 --dry-run 和 --prefer-binary 来检查是否有预编译包
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", f"numpy=={version}", "--dry-run", "--prefer-binary", "--only-binary", ":all:", "--report", "-"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # 检查输出中是否有 wheel 文件（预编译包）
        if "whl" in result.stdout or "wheel" in result.stdout.lower():
            print(f"  ✓ numpy {version} 可能有预编译包")
            return True
        else:
            print(f"  ✗ numpy {version} 没有找到预编译包")
            return False
    except Exception as e:
        print(f"  ✗ 检查失败: {e}")
        return False

def check_available_versions():
    """检查可用的 numpy 1.x 版本"""
    print("=" * 50)
    print("检查 numpy 1.x 版本在 Python 3.14 上的可用性")
    print("=" * 50)
    
    # 测试几个常见的 numpy 1.x 版本
    versions_to_check = [
        "1.26.4",  # 最新的 1.26.x
        "1.26.3",
        "1.26.2",
        "1.26.1",
        "1.26.0",
        "1.25.2",  # 最新的 1.25.x
        "1.25.1",
        "1.24.4",  # 最新的 1.24.x
    ]
    
    available_versions = []
    for version in versions_to_check:
        if check_numpy_version(version):
            available_versions.append(version)
    
    print("\n" + "=" * 50)
    if available_versions:
        print(f"找到 {len(available_versions)} 个可能有预编译包的版本:")
        for v in available_versions:
            print(f"  - numpy {v}")
        return available_versions[0]  # 返回最新的可用版本
    else:
        print("❌ 没有找到任何 numpy 1.x 版本的预编译包")
        return None

if __name__ == "__main__":
    print(f"Python 版本: {sys.version}")
    available = check_available_versions()
    if available:
        print(f"\n建议使用: numpy=={available}")
    else:
        print("\n建议: 使用 Python 3.11 或 3.12，或者安装编译工具")

