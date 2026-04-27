#!/bin/bash
# Databet 数据处理快速入门脚本

set -e  # 遇到错误立即退出

echo "=========================================="
echo "Databet 数据处理快速入门"
echo "=========================================="
echo ""

# 检查 Python 环境
echo "1. 检查 Python 环境..."
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 python3"
    exit 1
fi
echo "✓ Python 版本: $(python3 --version)"
echo ""

# 检查必要的 Python 包
echo "2. 检查必要的 Python 包..."
python3 -c "import torch; import cv2; import numpy; import tqdm" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "警告: 缺少必要的 Python 包"
    echo "请运行: pip install torch opencv-python numpy tqdm matplotlib"
    exit 1
fi
echo "✓ 所有必要的包已安装"
echo ""

# 检查输入数据
echo "3. 检查输入数据..."
if [ ! -d "databet" ]; then
    echo "错误: 未找到 databet 目录"
    echo "请确保 databet 目录存在并包含数据"
    exit 1
fi

# 统计序列数量
seq_count=$(find databet -mindepth 1 -maxdepth 1 -type d | wc -l)
echo "✓ 发现 $seq_count 个序列"
echo ""

# 运行数据处理
echo "4. 开始处理数据..."
echo "   这可能需要几分钟时间，请耐心等待..."
echo ""
python3 scripts/process_databet.py
if [ $? -ne 0 ]; then
    echo "错误: 数据处理失败"
    exit 1
fi
echo ""
echo "✓ 数据处理完成"
echo ""

# 运行测试
echo "5. 运行数据验证测试..."
echo ""
python3 scripts/test_data_processing.py
if [ $? -ne 0 ]; then
    echo "警告: 部分测试失败，请检查日志"
else
    echo ""
    echo "✓ 所有测试通过"
fi
echo ""

# 生成可视化
echo "6. 生成数据可视化..."
echo ""
python3 scripts/visualize_processed_data.py --mode sample --sample-idx 0
python3 scripts/visualize_processed_data.py --mode stats
echo ""
echo "✓ 可视化结果保存在 visualizations/ 目录"
echo ""

# 完成
echo "=========================================="
echo "🎉 快速入门完成！"
echo "=========================================="
echo ""
echo "下一步:"
echo "  1. 查看处理后的数据: processed_data/"
echo "  2. 查看可视化结果: visualizations/"
echo "  3. 阅读详细文档: docs/DATASET_PROCESSING_GUIDE.md"
echo "  4. 在训练中使用数据集"
echo ""
echo "示例代码:"
echo "  from viplanner.datasets.processed_databet_dataset import ProcessedDatabetDataset"
echo "  dataset = ProcessedDatabetDataset('processed_data', split='train')"
echo ""
