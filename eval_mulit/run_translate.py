#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
if os.path.isdir(_SRC_DIR) and _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


def _load_yaml(path: str) -> dict:
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        from omegaconf import OmegaConf
        return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def _load_test_data(data_dir: str, src_lang: str, tgt_lang: str) -> list[dict]:
    filename = f"{src_lang}_{tgt_lang}.json"
    path = os.path.join(data_dir, "translation_data", "test_data", filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"测试数据文件不存在: {path}\n"
            f"请确认 data/translation_data/test_data/{filename} 是否存在。"
        )
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    try:
        data = json.loads(content)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    records = []
    for line in content.splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser(
        description="统一翻译推理脚本（复用 LlamaFactory SFT 管线）"
    )
    parser.add_argument("--config", required=True,
                        help="YAML 配置文件路径（与训练 YAML 同格式）")
    parser.add_argument("--src_lang", required=True,
                        help="源语言码（en/zh/ru/de/cs/my）")
    parser.add_argument("--tgt_lang", required=True,
                        help="目标语言码（en/zh/ru/de/cs/my）")
    parser.add_argument("--out_file", required=True,
                        help="输出翻译结果 JSONL 路径")
    parser.add_argument("--prompt_save_dir", default=None,
                        help="覆盖 YAML 中的 prompt_save_dir（共享 encoder checkpoint 所在目录）")
    script_args = parser.parse_args()

    config = _load_yaml(script_args.config)

    test_dataset_name = f"test_{script_args.src_lang}_{script_args.tgt_lang}"
    config["eval_dataset"] = test_dataset_name
    if "dataset_dir" not in config:
        config["dataset_dir"] = os.path.join(_PROJECT_ROOT, "data")

    config["do_train"] = False
    config["do_eval"] = False
    config["do_predict"] = True
    config["predict_with_generate"] = True
    config["overwrite_output_dir"] = True

    lang_pair = f"{script_args.src_lang}-{script_args.tgt_lang}"
    config["current_lang_pair"] = lang_pair

    out_file_dir = os.path.dirname(os.path.abspath(script_args.out_file))
    config["output_dir"] = out_file_dir

    if script_args.prompt_save_dir:
        config["prompt_save_dir"] = script_args.prompt_save_dir

    if config.get("use_prompt_manager"):
        config["target_lang_pair"] = lang_pair
        config["current_forced_lang_pair"] = lang_pair

    config.setdefault("stage", "sft")

    if "RANK" not in os.environ:
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")

    print(f"[run_translate] 配置: {script_args.config}")
    print(f"[run_translate] 语言对: {lang_pair}")
    print(f"[run_translate] 测试集: {test_dataset_name}")
    print(f"[run_translate] 输出: {script_args.out_file}")

    os.environ["ALLOW_EXTRA_ARGS"] = "1" 

    from llamafactory.hparams import get_train_args
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(config)

    from llamafactory.train.sft.workflow import run_sft
    run_sft(model_args, data_args, training_args, finetuning_args, generating_args)

    data_dir = config.get("dataset_dir", os.path.join(_PROJECT_ROOT, "data"))
    original = _load_test_data(data_dir, script_args.src_lang, script_args.tgt_lang)

    pred_path = os.path.join(training_args.output_dir, "generated_predictions.jsonl")
    if not os.path.exists(pred_path):
        print(f"[run_translate] 错误: 预测结果文件不存在 {pred_path}")
        sys.exit(1)

    predictions = []
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                predictions.append(json.loads(line))

    if len(predictions) != len(original):
        print(
            f"[run_translate] 警告: 原始数据 {len(original)} 条 != 预测结果 {len(predictions)} 条，"
            f"将按较少的一方对齐。"
        )
    out_dir = os.path.dirname(script_args.out_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    n_written = 0
    with open(script_args.out_file, "w", encoding="utf-8") as f:
        for idx in range(min(len(original), len(predictions))):
            orig = original[idx]
            pred = predictions[idx]

            translation = pred.get("predict", "").strip()

            record = {
                "idx": idx,
                "src": orig.get("src", ""),
                "translation": translation,
                "ref": orig.get("ref", ""),
                "lang_pair": orig.get("lang_pair", lang_pair),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"[run_translate] ✓ 完成: {n_written} 条翻译结果已保存到 {script_args.out_file}")

    for _junk in ("all_results.json", "predict_results.json", "generated_predictions.jsonl"):
        _junk_path = os.path.join(training_args.output_dir, _junk)
        try:
            if os.path.exists(_junk_path):
                os.remove(_junk_path)
        except Exception:
            pass


if __name__ == "__main__":
    main()
