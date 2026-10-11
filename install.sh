#!/usr/bin/env bash
# 一键安装依赖（macOS / Linux / Git Bash）
# 用法：bash install.sh
set -e

PY=.venv/bin/python
if [ ! -x "$PY" ]; then
  echo "[1/4] 创建虚拟环境 .venv ..."
  python3 -m venv .venv
  PY=.venv/bin/python
fi

echo "[2/4] 升级 pip ..."
$PY -m pip install --upgrade pip

echo "[3/4] 安装 torch / torchvision（CPU 版）..."
$PY -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo "[4/4] 安装其余依赖 ..."
$PY -m pip install -r requirements.txt

echo "完成！在 VSCode 中选择解释器： .venv/bin/python"
