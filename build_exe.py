"""
小白AI图片去水印 - 打包脚本
使用 PyInstaller 将程序打包为 exe

使用方法:
1. 确保已安装依赖: pip install pyinstaller
2. 运行: python build_exe.py
3. 打包结果在 dist/小白AI图片去水印/ 目录
"""
import os
import sys
import shutil
import subprocess
from pathlib import Path

# 项目根目录
ROOT_DIR = Path(__file__).parent
DIST_DIR = ROOT_DIR / "dist"
BUILD_DIR = ROOT_DIR / "build"
OUTPUT_NAME = "小白AI图片去水印"


def clean_build():
    """清理之前的构建文件"""
    print("🧹 清理旧的构建文件...")
    for dir_path in [DIST_DIR, BUILD_DIR]:
        if dir_path.exists():
            shutil.rmtree(dir_path)
    
    # 删除 spec 文件
    for spec_file in ROOT_DIR.glob("*.spec"):
        spec_file.unlink()


def create_version_file():
    """创建版本信息文件（减少杀毒误报）"""
    version_content = '''
VSVersionInfo(
  ffi=FixedFileInfo(
    filevers=(1, 0, 0, 0),
    prodvers=(1, 0, 0, 0),
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo([
      StringTable(
        u'080404b0',
        [StringStruct(u'CompanyName', u'小白AI'),
         StringStruct(u'FileDescription', u'小白AI图片去水印工具'),
         StringStruct(u'FileVersion', u'1.0.0'),
         StringStruct(u'InternalName', u'xiaobai_watermark'),
         StringStruct(u'LegalCopyright', u'Copyright (C) 2025'),
         StringStruct(u'OriginalFilename', u'小白AI图片去水印.exe'),
         StringStruct(u'ProductName', u'小白AI图片去水印'),
         StringStruct(u'ProductVersion', u'1.0.0')])
    ]),
    VarFileInfo([VarStruct(u'Translation', [2052, 1200])])
  ]
)
'''
    version_file = ROOT_DIR / "version_info.txt"
    version_file.write_text(version_content, encoding='utf-8')
    return version_file


def build_exe():
    """执行打包"""
    print("📦 开始打包...")
    
    # 创建版本信息文件
    version_file = create_version_file()
    
    # PyInstaller 参数
    # 注意: 不使用 --onefile 和 UPX，减少杀毒误报
    # 使用 --console 显示控制台，避免 stdin/stdout 问题
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", OUTPUT_NAME,
        "--console",  # 显示控制台窗口，用户可以看到日志
        "--noconfirm",
        "--clean",
        "--noupx",  # 不使用 UPX 压缩，减少误报
        f"--version-file={version_file}",
        
        # 添加数据文件
        "--add-data", f"index.html{os.pathsep}.",
        "--add-data", f"main.py{os.pathsep}.",
        
        # 隐藏导入
        "--hidden-import", "uvicorn.logging",
        "--hidden-import", "uvicorn.loops",
        "--hidden-import", "uvicorn.loops.auto",
        "--hidden-import", "uvicorn.protocols",
        "--hidden-import", "uvicorn.protocols.http",
        "--hidden-import", "uvicorn.protocols.http.auto",
        "--hidden-import", "uvicorn.protocols.websockets",
        "--hidden-import", "uvicorn.protocols.websockets.auto",
        "--hidden-import", "uvicorn.lifespan",
        "--hidden-import", "uvicorn.lifespan.on",
        
        # 入口文件
        "launcher.py"
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, cwd=ROOT_DIR)
    
    if result.returncode != 0:
        print("❌ 打包失败!")
        return False
    
    return True


def copy_resources():
    """复制资源文件到输出目录"""
    print("📂 复制资源文件...")
    
    output_dir = DIST_DIR / OUTPUT_NAME
    
    # 复制 index.html
    shutil.copy(ROOT_DIR / "index.html", output_dir / "index.html")
    
    # 复制 main.py（作为模块导入）
    shutil.copy(ROOT_DIR / "main.py", output_dir / "main.py")
    
    # 复制模型文件（如果存在）
    models_src = ROOT_DIR / "models"
    if models_src.exists():
        models_dst = output_dir / "models"
        if models_dst.exists():
            shutil.rmtree(models_dst)
        shutil.copytree(models_src, models_dst)
        print(f"  ✓ 复制模型文件: {models_dst}")
    
    # 创建空目录
    (output_dir / "uploads").mkdir(exist_ok=True)
    (output_dir / "outputs").mkdir(exist_ok=True)
    
    print(f"  ✓ 资源文件已复制到: {output_dir}")


def create_readme():
    """创建使用说明"""
    readme_content = """# 小白AI图片去水印

## 使用方法

1. 双击运行 `小白AI图片去水印.exe`
2. 程序会自动打开浏览器
3. 上传图片即可去除水印

## 注意事项

- 首次运行可能需要加载 AI 模型，请耐心等待
- 如果杀毒软件误报，请添加信任
- 关闭程序窗口将停止服务

## 文件说明

- `小白AI图片去水印.exe` - 主程序
- `models/` - AI 模型文件
- `uploads/` - 上传文件临时目录
- `outputs/` - 输出文件目录

## 问题反馈

如有问题，请在 GitHub Issues 中反馈。
"""
    output_dir = DIST_DIR / OUTPUT_NAME
    readme_file = output_dir / "使用说明.txt"
    readme_file.write_text(readme_content, encoding='utf-8')
    print(f"  ✓ 创建使用说明: {readme_file}")


def main():
    print("=" * 60)
    print("✨ 小白AI图片去水印 - 打包工具")
    print("=" * 60)
    
    # 检查 PyInstaller
    try:
        import PyInstaller
        print(f"✓ PyInstaller 版本: {PyInstaller.__version__}")
    except ImportError:
        print("❌ 请先安装 PyInstaller: pip install pyinstaller")
        sys.exit(1)
    
    # 清理
    clean_build()
    
    # 打包
    if not build_exe():
        sys.exit(1)
    
    # 复制资源
    copy_resources()
    
    # 创建说明文件
    create_readme()
    
    # 清理临时文件
    version_file = ROOT_DIR / "version_info.txt"
    if version_file.exists():
        version_file.unlink()
    
    print("=" * 60)
    print("✅ 打包完成!")
    print(f"📁 输出目录: {DIST_DIR / OUTPUT_NAME}")
    print("=" * 60)
    print("\n提示: 可以将整个目录压缩后分发")


if __name__ == "__main__":
    main()

