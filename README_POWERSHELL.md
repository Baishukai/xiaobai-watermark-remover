# PowerShell 执行策略问题解决方案

如果你在 PowerShell 中遇到执行策略错误，有以下几种解决方法：

## 方法1：使用 PowerShell 脚本（推荐）

我们提供了 PowerShell 版本的脚本：

```powershell
# 安装（需要管理员权限或设置执行策略）
.\setup.ps1

# 启动
.\start.ps1
```

## 方法2：临时允许脚本执行

在当前 PowerShell 会话中临时允许脚本执行：

```powershell
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
.\setup.ps1
```

## 方法3：使用 CMD 命令行

直接使用 CMD 而不是 PowerShell：

1. 按 `Win + R`
2. 输入 `cmd` 并按回车
3. 在 CMD 中运行：
   ```cmd
   cd F:\CursorCode_all\图片去水印
   setup.bat
   ```

## 方法4：永久更改执行策略（需要管理员权限）

以管理员身份运行 PowerShell，然后执行：

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

这会将当前用户的执行策略设置为 `RemoteSigned`，允许运行本地脚本。

## 方法5：直接使用 Python 命令

如果虚拟环境已创建，可以直接使用：

```powershell
# 激活虚拟环境（如果执行策略允许）
.\venv\Scripts\Activate.ps1

# 或者直接使用完整路径
.\venv\Scripts\python.exe main.py
```

## 推荐方案

对于开发环境，建议使用方法4（设置执行策略为 RemoteSigned），这样既安全又方便。

对于生产环境部署，建议使用方法3（使用 CMD）或直接使用 Python 命令。

