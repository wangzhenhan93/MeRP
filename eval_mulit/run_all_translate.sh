#!/bin/bash

set -e

CONFIG=$1
RESULT_DIR=$2

if [ -z "$CONFIG" ] || [ -z "$RESULT_DIR" ]; then
    echo "用法: bash $0 <CONFIG_YAML> <RESULT_DIR>"
    echo "示例: bash $0 eval_mulit/configs/sda_nm_gemma2.yaml results/sda_nm_gemma2/full"
    exit 1
fi

PAIRS=()
if [ -n "$3" ]; then
    IFS=',' read -r -a PAIRS <<< "$3"
else
    if command -v python >/dev/null 2>&1; then
        mapfile -t PAIRS < <(python - "$CONFIG" <<'PY'
import re
import sys

cfg = sys.argv[1]
text = open(cfg, 'r', encoding='utf-8').read()

pairs = []
try:
    import yaml  # type: ignore
    data = yaml.safe_load(text) or {}
    raw = data.get('lang_pairs', [])
    if isinstance(raw, str):
        raw = [raw]
    if isinstance(raw, list):
        pairs = [str(x).strip().strip('"').strip("'") for x in raw if str(x).strip()]
except Exception:
    pairs = []

if not pairs:
    m = re.search(r"^\s*lang_pairs\s*:\s*(.*)$", text, flags=re.M)
    if m:
        tail = (m.group(1) or '').strip()
        if tail.startswith('[') and tail.endswith(']'):
            inner = tail[1:-1]
            for token in inner.split(','):
                tok = token.strip().strip('"').strip("'")
                if tok:
                    pairs.append(tok)
        else:
            lines = text[m.end():].splitlines()
            for line in lines:
                if not line.strip():
                    continue
                if not re.match(r"^\s*-\s*", line):
                    break
                tok = re.sub(r"^\s*-\s*", '', line).strip().strip('"').strip("'")
                if tok:
                    pairs.append(tok)

for p in pairs:
    print(p)
PY
)
    fi
fi

if [ ${#PAIRS[@]} -eq 0 ]; then
    echo "[警告] 未能从配置解析 lang_pairs，回退到默认 10 个方向。"
    PAIRS=("ru-en" "en-ru" "de-en" "en-de" "cs-en" "en-cs" "en-zh" "zh-en" "vi-en" "en-vi" )
fi


mkdir -p "${RESULT_DIR}"

echo "=========================================="
echo " 配置: ${CONFIG}"
echo " 输出: ${RESULT_DIR}"
echo " 语言对: ${PAIRS[*]}"
echo "=========================================="

for PAIR in "${PAIRS[@]}"; do
    IFS='-' read -r SRC TGT <<< "$PAIR"

    echo ""
    echo "=== [${PAIR}] 开始翻译 ==="
    python eval_mulit/run_translate.py \
        --config "${CONFIG}" \
        --src_lang "${SRC}" \
        --tgt_lang "${TGT}" \
        --out_file "${RESULT_DIR}/${PAIR}/${PAIR}_translations.jsonl"
    echo "=== [${PAIR}] 翻译完成 ==="
done

echo ""
echo "=========================================="
echo " 全部翻译完成！结果目录: ${RESULT_DIR}"
echo "=========================================="
