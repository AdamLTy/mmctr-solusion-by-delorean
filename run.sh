#!/bin/bash

# 定义目录和文件路径
REQUIREMENTS_FILE="requirements.txt"
LOCAL_PACKAGE_DIR="transformers-main"  # 替换为本地包目录

# 定义要解压的zip文件名
ZIP_FILE="transformers-main.zip"

# 检查zip文件是否存在
if [ ! -f "$ZIP_FILE" ]; then
    echo "错误：文件 $ZIP_FILE 不存在"
    exit 1
fi

# 解压文件到当前目录
echo "正在解压 $ZIP_FILE 到当前目录..."
unzip -o "$ZIP_FILE"

echo "解压完成！"

# 安装本地包
echo "正在安装本地包..."
cd $LOCAL_PACKAGE_DIR
pip install .

# 检查requirements.txt文件是否存在
cd ..
if [ -f "$REQUIREMENTS_FILE" ]; then
    echo "正在通过requirements.txt安装依赖..."
    pip install -r $REQUIREMENTS_FILE
else
    echo "错误: requirements.txt文件不存在于 $REQUIREMENTS_FILE"
    exit 1
fi

echo "安装完成!"


# 定义Python文件路径
PYTHON_FILE1="inference.py"
PYTHON_FILE2="PCA.py"
PYTHON_FILE3="gen_item_info.py"

# 检查Python是否已安装
if ! command -v python3 &> /dev/null; then
    echo "Python3未找到，请安装Python3后再试"
    exit 1
fi

# 运行第一个Python文件
echo "正在运行第一个Python脚本: $PYTHON_FILE1"
python3 $PYTHON_FILE1
# 检查上一个命令是否成功执行
if [ $? -ne 0 ]; then
    echo "第一个Python脚本执行失败"
    exit 1
fi

# 运行第二个Python文件
echo "正在运行第二个Python脚本: $PYTHON_FILE2"
python3 $PYTHON_FILE2
if [ $? -ne 0 ]; then
    echo "第二个Python脚本执行失败"
    exit 1
fi

# 运行第三个Python文件
echo "正在运行第三个Python脚本: $PYTHON_FILE3"
python3 $PYTHON_FILE3
if [ $? -ne 0 ]; then
    echo "第三个Python脚本执行失败"
    exit 1
fi

# 定义变量（可选，但使脚本更易于维护）
PYTHON_SCRIPT="run_param_tuner.py"
CONFIG_FILE="config/DIN_microlens_mmctr_tuner_config_01.yaml"
GPU_ID=0

# 运行Python脚本并传递参数
echo "正在GPU ${GPU_ID}上运行参数调优脚本..."
python $PYTHON_SCRIPT --config $CONFIG_FILE --gpu $GPU_ID

# 检查命令是否成功执行
if [ $? -eq 0 ]; then
    echo "参数调优脚本执行成功！"
else
    echo "参数调优脚本执行失败，错误代码: $?"
    exit 1
fi

echo "所有Python脚本已成功执行完毕！"
