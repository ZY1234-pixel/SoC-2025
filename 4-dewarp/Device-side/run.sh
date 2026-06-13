#!/usr/bin/env bash
set -e

echo "============================"
echo " Build DeepLab NCNN Project"
echo "============================"

# 回到脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 清理旧文件
rm -f main

# ============================
# 编译
# ============================
g++ main.cpp deeplabv3p_ncnn.cpp -o main \
-std=c++17 \
-Incnn-20260113/build/install/include \
-Lncnn-20260113/build/install/lib \
-lncnn \
$(pkg-config --cflags --libs opencv4) \
-fopenmp -lgomp

echo "============================"
echo " Build Done, Running..."
echo "============================"

./main
