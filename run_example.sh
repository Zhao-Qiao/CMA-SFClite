#!/bin/bash
# 运行 NIE - Feature Ablation 实验的示例脚本

echo "======================================"
echo "NIE - Feature Ablation 实验"
echo "整合 CMA 和 SFC 方法"
echo "======================================"
echo ""

# 配置
MODEL="gpt2"  # HuggingFace上的标准名称（对应GPT2-Small）
TOPK_JSON="experiments/data/topk_mediators_gpt2-small_doctor_woman_20251026_150239.json"
EVAL_DATA="experiments/data/bias_eval_gpt2-small_doctor_woman_20251026_150239.json"
OUTPUT_DIR="results"
DEVICE="cpu"  # 或 "cpu"

# 检查 CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "✓ 检测到 CUDA，将使用 GPU"
    DEVICE="cuda"
else
    echo "⚠ 未检测到 CUDA，将使用 CPU"
    DEVICE="cpu"
fi

# 运行实验
echo ""
echo "开始运行实验..."
echo "- 模型: $MODEL"
echo "- Top-K Mediators: $TOPK_JSON"
echo "- 评估数据: $EVAL_DATA"
echo "- 输出目录: $OUTPUT_DIR"
echo "- 设备: $DEVICE"
echo ""

python experiments/nie_feature_ablation.py \
  --model "$MODEL" \
  --topk_json "$TOPK_JSON" \
  --eval_data "$EVAL_DATA" \
  --output_dir "$OUTPUT_DIR" \
  --topk 3 \
  --num_features 5 \
  --gating_target both \
  --max_examples 3 \
  --random_baseline \
  --include_head_off \
  --control_mediators 2 \
  --nie_thresholds "0.75,0.5,0.25" \
  --seed 0 \
  --device "$DEVICE"

echo ""
echo "======================================"
echo "实验完成！"
echo "结果保存在: $OUTPUT_DIR"
echo "======================================"
