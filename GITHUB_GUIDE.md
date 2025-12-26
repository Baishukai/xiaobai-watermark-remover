# GitHub 仓库创建和发布指南

## 一、创建 GitHub 仓库

### 1.1 在 GitHub 网页上创建仓库

1. 登录 GitHub: https://github.com
2. 点击右上角 "+" → "New repository"
3. 填写仓库信息：
   - Repository name: `xiaobai-watermark-remover`
   - Description: `小白AI图片去水印 - 简单易用的本地图片水印去除工具`
   - 选择 Public（公开）
   - 不要勾选 "Add a README file"（我们已有）
4. 点击 "Create repository"

### 1.2 本地初始化并推送

```powershell
# 进入项目目录
cd F:\CursorCode_all\图片去水印

# 初始化 Git 仓库
git init

# 创建 .gitignore 文件（排除不需要上传的文件）
# 文件内容见下方

# 添加所有文件
git add .

# 提交
git commit -m "Initial commit: 小白AI图片去水印 v1.0"

# 添加远程仓库（替换 YOUR_USERNAME 为你的 GitHub 用户名）
git remote add origin https://github.com/YOUR_USERNAME/xiaobai-watermark-remover.git

# 推送到 GitHub
git branch -M main
git push -u origin main
```

## 二、.gitignore 文件内容

创建 `.gitignore` 文件，内容如下：

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
dist/
*.egg-info/
*.egg

# 虚拟环境
venv/
.venv/
ENV/

# IDE
.idea/
.vscode/
*.swp
*.swo

# 临时文件
uploads/*
outputs/*
!uploads/.gitkeep
!outputs/.gitkeep

# 模型文件（太大，单独下载）
models/*.pt

# 构建产物
*.spec
version_info.txt

# 测试生成的图片
test_*.png
compare_*.png
diff_*.png
corner_*.png
debug_*.png
*_result.png
processed-image.png
```

## 三、发布 Release（包含 exe）

### 3.1 打包 exe

```powershell
# 激活虚拟环境
.\venv\Scripts\Activate.ps1

# 安装 PyInstaller
pip install pyinstaller

# 运行打包脚本
python build_exe.py
```

打包完成后，在 `dist/小白AI图片去水印/` 目录下会生成可执行文件。

### 3.2 压缩打包目录

```powershell
# 压缩整个目录
Compress-Archive -Path "dist\小白AI图片去水印\*" -DestinationPath "小白AI图片去水印_v1.0.zip"
```

### 3.3 在 GitHub 创建 Release

1. 打开你的 GitHub 仓库页面
2. 点击右侧 "Releases" → "Create a new release"
3. 填写信息：
   - Tag version: `v1.0.0`
   - Release title: `小白AI图片去水印 v1.0.0`
   - Description:
     ```
     ## 小白AI图片去水印 v1.0.0
     
     ### 功能特性
     - ✨ 自动检测水印位置
     - 🎯 可拖拽调整水印区域
     - 🖌️ 手动标记模式
     - 🤖 LaMa AI 智能去水印
     
     ### 使用方法
     1. 下载 `小白AI图片去水印_v1.0.zip`
     2. 解压到任意目录
     3. 双击运行 `小白AI图片去水印.exe`
     4. 浏览器会自动打开，上传图片即可
     
     ### 注意事项
     - 首次运行需要下载 AI 模型（约 200MB）
     - 如杀毒软件误报，请添加信任
     ```
4. 上传压缩包：拖拽 `小白AI图片去水印_v1.0.zip` 到 "Attach binaries" 区域
5. 点击 "Publish release"

## 四、模型文件说明

由于 LaMa 模型文件较大（约 200MB），有两种处理方式：

### 方式1：首次运行自动下载（默认）

程序会在首次运行时自动从网络下载模型，存放在用户目录。

### 方式2：预置模型文件

如果需要离线使用，可以：
1. 下载模型文件：`big-lama.pt`
2. 放置到 `dist/小白AI图片去水印/models/` 目录
3. 重新打包 zip

模型下载地址：
https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt

## 五、常见问题

### Q: 杀毒软件误报怎么办？

PyInstaller 打包的 exe 有时会被误报。解决方法：
1. 使用 `--noupx` 参数（已在脚本中使用）
2. 不使用 `--onefile` 模式
3. 添加版本信息
4. 用户手动添加信任/白名单

### Q: exe 文件太大怎么办？

由于包含了 Python 解释器和依赖库，打包后体积较大（约 500MB+）。
可以考虑：
1. 使用虚拟环境精简依赖
2. 不打包 exe，提供源码运行方式
3. 使用 Docker 部署

---

*指南完成，祝发布顺利！*

