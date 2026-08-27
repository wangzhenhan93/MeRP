#!/bin/bash

set -e

RESULT_DIR=$1
COMET_MODEL=$2
LOCAL_XLM=$3     

if [ -z "$RESULT_DIR" ]; then
    echo "用法: bash $0 <RESULT_DIR> [COMET_MODEL] [LOCAL_XLM_PATH]"
    echo "示例: bash $0 results/lart /path/to/wmt22-comet-da /path/to/xlm-roberta-large"
    exit 1
fi

PAIRS=()
for pair_dir in "${RESULT_DIR}"/*; do
    [ -d "${pair_dir}" ] || continue
    pair_name=$(basename "${pair_dir}")
    if [ -f "${pair_dir}/${pair_name}_translations.jsonl" ]; then
        PAIRS+=("${pair_name}")
    fi
done

if [ ${#PAIRS[@]} -eq 0 ]; then
    echo "[警告] 在 ${RESULT_DIR} 下未发现任何 *_translations.jsonl，回退到默认 10 个方向。"
    PAIRS=("ru-en" "en-ru" "de-en" "en-de" "cs-en" "en-cs" "en-zh" "zh-en" "vi-en" "en-vi" )
fi


echo "=========================================="
echo " 评估目录: ${RESULT_DIR}"
echo " COMET 模型: ${COMET_MODEL:-未指定（跳过 COMET）}"
echo " XLM-R 路径: ${LOCAL_XLM:-未指定}"
echo " 语言对: ${PAIRS[*]}"
echo "=========================================="

for PAIR in "${PAIRS[@]}"; do
    IFS='-' read -r SRC TGT <<< "$PAIR"

    TRANS_FILE="${RESULT_DIR}/${PAIR}/${PAIR}_translations.jsonl"
    if [ ! -f "$TRANS_FILE" ]; then
        echo "[${PAIR}] 翻译文件不存在: ${TRANS_FILE}，跳过。"
        continue
    fi

    echo ""
    echo "=== [${PAIR}] 开始评估 ==="

    COMET_ARGS=""
    if [ -n "$COMET_MODEL" ]; then
        COMET_ARGS="--comet_model ${COMET_MODEL}"
    fi
    if [ -n "$LOCAL_XLM" ]; then
        COMET_ARGS="${COMET_ARGS} --local_xlm_path ${LOCAL_XLM}"
    fi

    python eval_mulit/evaluate_new.py \
        --translations "${TRANS_FILE}" \
        --tgt_lang "${TGT}" \
        --metrics_output "${RESULT_DIR}/${PAIR}/${PAIR}_scores.json" \
        --per_sample_output "${RESULT_DIR}/${PAIR}/${PAIR}_per_sample.json" \
        ${COMET_ARGS}

    echo "=== [${PAIR}] 评估完成 ==="
done

echo ""
echo "=========================================="
echo " 全部评估完成！"
echo ""

echo " ---- 指标汇总 ----"
for PAIR in "${PAIRS[@]}"; do
    SCORE_FILE="${RESULT_DIR}/${PAIR}/${PAIR}_scores.json"
    if [ -f "$SCORE_FILE" ]; then
        echo " [${PAIR}] $(cat ${SCORE_FILE})"
    fi
done
echo "=========================================="
