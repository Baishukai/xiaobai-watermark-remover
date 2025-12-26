# LaMa 模型文件

此目录存放 LaMa (Large Mask Inpainting) AI 模型文件。

## 文件说明

| 文件 | 大小 | 说明 |
|------|------|------|
| `big-lama.pt` | ~196MB | LaMa 模型权重文件 |

## 模型来源

- **项目**: [simple-lama-inpainting](https://github.com/enesmsahin/simple-lama-inpainting)
- **下载地址**: https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt

## 部署说明

打包部署时，请确保包含此目录及 `big-lama.pt` 文件。

如果模型文件丢失，程序会自动从网络下载（需要网络连接）。

## 手动下载

如果需要手动下载模型，可以使用以下命令：

```powershell
# PowerShell
Invoke-WebRequest -Uri "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt" -OutFile "models/big-lama.pt"
```

```bash
# Linux/macOS
curl -L -o models/big-lama.pt https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt
```

