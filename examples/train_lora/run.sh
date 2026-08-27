#!/bin/bash
# ═══════════════════════════════════════════════
# 完整推理 + 评估流水线
# chmod +x examples/train_lora/run.sh && ./examples/train_lora/run.sh
# ═══════════════════════════════════════════════
COMET_MODEL="/xxxxxxxxxx/wmt22-comet-da/checkpoints/model.ckpt"
LOCAL_XLM="/xxxxxxxxxxx/xlm-roberta-large"

(

  PYTHONPATH=/xxxxxxxxxx/LLaMA-Factory-2/src FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=7 llamafactory-cli train examples/train_lora/gemma2_2b_prompt_multi.yaml  && \

  CUDA_VISIBLE_DEVICES=7 MASTER_PORT=29508 \
  bash eval_mulit/run_all_translate.sh \
    eval_mulit/configs/merp_gemma2.yaml \
    results/gemma2 \
  &&

  CUDA_VISIBLE_DEVICES=7 \
  bash eval_mulit/run_all_evaluate.sh \
    results/gemma2 \
    "$COMET_MODEL" "$LOCAL_XLM" \

  echo "[gemma] 完成" 

) &
PID_GROUP_A=$!

wait $PID_GROUP_A

echo "=========================================="
echo " 全部完成！
echo "=========================================="







