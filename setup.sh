#!/bin/bash
echo "正在创建虚拟环境..."
python3 -m venv venv

echo "正在激活虚拟环境..."
source venv/bin/activate

echo "正在升级pip..."
pip install --upgrade pip

echo "正在安装依赖包..."
pip install -r requirements.txt

echo ""
echo "========================================"
echo "安装完成！"
echo "========================================"
echo ""
echo "运行 ./start.sh 启动服务"
echo "或者运行: python main.py"
echo ""

