@echo off
echo 正在创建虚拟环境...
python -m venv venv

echo 正在激活虚拟环境...
call venv\Scripts\activate.bat

REM 如果在PowerShell中运行，使用以下命令替代上面的call命令:
REM cmd /c "venv\Scripts\activate.bat && python -m pip install --upgrade pip && pip install -r requirements.txt"

echo 正在升级pip...
python -m pip install --upgrade pip

echo 正在安装依赖包...
pip install -r requirements.txt

echo.
echo ========================================
echo 安装完成！
echo ========================================
echo.
echo 运行 start.bat 启动服务
echo 或者运行: python main.py
echo.
pause

