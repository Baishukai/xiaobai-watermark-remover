# 图片去水印工具

一个基于FastAPI和HTML的在线图片去水印工具，支持去除图片右下角的水印。

## 功能特性

- ✅ 支持文件上传
- ✅ 支持拖拽上传
- ✅ 支持粘贴上传（Ctrl+V）
- ✅ 可调节水印区域大小
- ✅ 支持API并发调用
- ✅ 美观的现代化UI界面

## 技术栈

- **后端**: FastAPI
- **前端**: HTML + CSS + JavaScript
- **图像处理**: OpenCV + Pillow
- **部署**: Uvicorn

## 快速开始

### Windows系统

#### 使用 CMD（推荐，兼容性最好）

1. **打开 CMD**（按 Win+R，输入 `cmd`）

2. **一键安装**:
   ```cmd
   cd 图片去水印
   setup.bat
   ```

3. **启动服务**:
   ```cmd
   start.bat
   ```

#### 使用 PowerShell

**方法A：使用快速脚本（推荐）**

1. **右键点击 `快速安装.ps1` -> 使用 PowerShell 运行**
2. **安装完成后，右键点击 `快速启动.ps1` -> 使用 PowerShell 运行**

**方法B：手动执行**

如果遇到执行策略错误，先执行：
```powershell
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
```

然后运行：
```powershell
cd 图片去水印
.\setup.ps1
.\start.ps1
```

更多解决方案请查看 `README_POWERSHELL.md`

#### 直接使用 Python

如果虚拟环境已创建：
```bash
venv\Scripts\activate
python main.py
```

### Linux/Mac系统

1. **一键安装**（推荐）:
   ```bash
   cd 图片去水印
   chmod +x setup.sh start.sh
   ./setup.sh
   ```

2. **启动服务**:
   ```bash
   ./start.sh
   ```
   或直接运行:
   ```bash
   python main.py
   ```

### 手动安装步骤

如果不使用一键安装脚本，可以手动执行：

**Windows**:
```bash
cd 图片去水印
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

**Linux/Mac**:
```bash
cd 图片去水印
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

## 运行项目

### 开发模式

```bash
python main.py
```

或者使用uvicorn直接运行：

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 生产模式（Linux部署）

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

## API接口

### 去水印接口

**POST** `/api/remove-watermark`

**请求参数**:
- `file`: 图片文件（multipart/form-data）
- `watermark_ratio`: 水印区域比例（可选，默认0.15，范围0.1-0.3）

**响应**: 处理后的图片文件流

**示例**:

```bash
curl -X POST "http://localhost:8000/api/remove-watermark" \
  -F "file=@test.jpg" \
  -F "watermark_ratio=0.15" \
  --output result.jpg
```

### 健康检查接口

**GET** `/health`

返回服务状态信息。

## 使用说明

1. 打开浏览器访问 `http://localhost:8000`
2. 点击上传区域或拖拽图片到上传区域
3. 也可以直接按Ctrl+V粘贴剪贴板中的图片
4. 调整水印区域大小滑块（如果水印区域较大或较小）
5. 点击"开始去水印"按钮
6. 等待处理完成，预览结果
7. 点击"下载图片"保存处理后的图片

## 项目结构

```
图片去水印/
├── main.py              # FastAPI后端主程序
├── index.html           # 前端页面
├── requirements.txt     # Python依赖包
├── README.md           # 项目说明文档
├── API_EXAMPLE.md      # API使用示例
├── setup.bat           # Windows CMD安装脚本
├── setup.ps1           # Windows PowerShell安装脚本
├── setup.sh            # Linux/Mac一键安装脚本
├── start.bat           # Windows CMD启动脚本
├── start.ps1           # Windows PowerShell启动脚本
├── start.sh            # Linux/Mac启动脚本
├── README_POWERSHELL.md # PowerShell使用说明
├── test_api.py         # API测试脚本
├── .gitignore          # Git忽略文件
├── venv/               # Python虚拟环境（运行setup后生成）
├── uploads/            # 上传文件目录（自动创建）
└── outputs/            # 输出文件目录（自动创建）
```

## 测试API

使用提供的测试脚本：

```bash
# 安装测试依赖
pip install requests

# 运行测试
python test_api.py 右下角有水印.png

# 或指定输出文件名
python test_api.py 右下角有水印.png result.png
```

更多API使用示例请查看 `API_EXAMPLE.md` 文件。

## Linux部署说明

### 1. 上传项目到Linux服务器

```bash
scp -r 图片去水印 user@server:/path/to/deploy/
```

### 2. 在服务器上创建虚拟环境并安装依赖

```bash
cd /path/to/deploy/图片去水印
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. 使用systemd创建服务（可选）

创建服务文件 `/etc/systemd/system/watermark-remover.service`:

```ini
[Unit]
Description=Watermark Remover Service
After=network.target

[Service]
User=your_user
WorkingDirectory=/path/to/deploy/图片去水印
Environment="PATH=/path/to/deploy/图片去水印/venv/bin"
ExecStart=/path/to/deploy/图片去水印/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4

[Install]
WantedBy=multi-user.target
```

启动服务：

```bash
sudo systemctl daemon-reload
sudo systemctl enable watermark-remover
sudo systemctl start watermark-remover
```

### 4. 使用Nginx反向代理（可选）

在Nginx配置中添加：

```nginx
server {
    listen 80;
    server_name your_domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        client_max_body_size 50M;
    }
}
```

## 注意事项

- 处理大图片时可能需要较长时间
- 建议在生产环境中使用反向代理和负载均衡
- 上传的文件会临时存储在内存中处理，不会持久化到磁盘
- 水印去除效果取决于水印的位置、大小和背景复杂度

## 许可证

MIT License

