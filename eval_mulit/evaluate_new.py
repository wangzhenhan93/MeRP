#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import argparse
import logging
import shutil
from typing import Optional
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("evaluate")


def read_jsonl(path: str) -> list[dict]:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")

    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        return []
    try:
        data = json.loads(content)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
    except json.JSONDecodeError:
        pass

    records = []
    for line in content.splitlines():
        line = line.strip()
        if not line or line in ("[", "]"):
            continue
        if line.endswith(","):
            line = line[:-1].rstrip()
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def write_json(path: str, obj) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _get_bleu_tokenizer(tgt_lang: str) -> str:
    lang = (tgt_lang or "").strip().lower()

    if lang == "zh":
        return "zh"

    if lang in ("my", "th", "lo", "km", "vi"):
        return "flores200"
    return "13a"


def compute_sentence_metrics(
    hyp: str, ref: str, src: str, tgt_lang: str
) -> dict[str, float]:
    import sacrebleu

    bleu_tok = _get_bleu_tokenizer(tgt_lang)

    bleu = sacrebleu.sentence_bleu(hyp, [ref], tokenize=bleu_tok).score

    sp_bleu = sacrebleu.sentence_bleu(hyp, [ref], tokenize="flores200").score

    chrf = sacrebleu.sentence_chrf(hyp, [ref], word_order=0).score

    chrfpp = sacrebleu.sentence_chrf(hyp, [ref], word_order=2).score

    return {
        "bleu": round(float(bleu), 4),
        "sp_bleu": round(float(sp_bleu), 4),
        "chrf": round(float(chrf), 4),
        "chrfpp": round(float(chrfpp), 4),
    }


def compute_corpus_metrics(
    hyps: list[str], refs: list[str], tgt_lang: str
) -> dict[str, float]:
    import sacrebleu

    if not hyps or not refs:
        return {"BLEU": 0.0, "spBLEU": 0.0, "chrF": 0.0, "chrF++": 0.0}

    bleu_tok = _get_bleu_tokenizer(tgt_lang)

    bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize=bleu_tok).score
    sp_bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize="flores200").score
    chrf = sacrebleu.corpus_chrf(hyps, [refs], word_order=0).score
    chrfpp = sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score

    return {
        "BLEU": round(float(bleu), 2),
        "spBLEU": round(float(sp_bleu), 2),
        "chrF": round(float(chrf), 2),
        "chrF++": round(float(chrfpp), 2),
    }


def _apply_xlmr_monkeypatch(local_xlm_dir: str) -> None:
    try:
        import transformers as _tf
    except ImportError:
        logger.warning("transformers 未安装，无法应用 xlm-roberta-large 本地重定向。")
        return

    def _is_xlmr(name: str) -> bool:
        return isinstance(name, str) and "xlm-roberta-large" in name

    for cls_name in [
        "AutoTokenizer", "AutoModel", "PretrainedConfig",
        "XLMRobertaTokenizerFast", "XLMRobertaModel", "XLMRobertaForMaskedLM",
    ]:
        cls = getattr(_tf, cls_name, None)
        if cls is None or not hasattr(cls, "from_pretrained"):
            continue
        _orig = cls.from_pretrained

        def _make_patched(original):
            def _patched(name_or_path, *args, **kwargs):
                if _is_xlmr(str(name_or_path)):
                    kwargs.setdefault("local_files_only", True)
                    return original(local_xlm_dir, *args, **kwargs)
                return original(name_or_path, *args, **kwargs)
            return _patched

        cls.from_pretrained = _make_patched(_orig)

    logger.info("已将 xlm-roberta-large 重定向到本地: %s", local_xlm_dir)


def _prepare_hf_snapshot(local_xlm_dir: str) -> None:
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    hub_dir = os.path.join(hf_home, "hub", "models--xlm-roberta-large", "snapshots")
    os.makedirs(hub_dir, exist_ok=True)
    link = os.path.join(hub_dir, "local_offline")

    if os.path.islink(link) or os.path.exists(link):
        try:
            if os.path.islink(link):
                os.unlink(link)
            elif os.path.isdir(link):
                shutil.rmtree(link)
            else:
                os.remove(link)
        except Exception:
            pass

    try:
        os.symlink(os.path.abspath(local_xlm_dir), link)
    except (OSError, NotImplementedError):
        shutil.copytree(os.path.abspath(local_xlm_dir), link)


def load_comet_model(
    comet_model_path: Optional[str],
    local_xlm_path: Optional[str],
):
    if local_xlm_path:
        local_xlm_path = os.path.abspath(local_xlm_path)
        if not os.path.exists(local_xlm_path):
            logger.warning("local_xlm_path 不存在: %s，跳过 COMET。", local_xlm_path)
            return None
        _prepare_hf_snapshot(local_xlm_path)
        _apply_xlmr_monkeypatch(local_xlm_path)

    try:
        from comet import load_from_checkpoint
    except ImportError:
        logger.warning("unbabel-comet 未安装，跳过 COMET 评估。pip install unbabel-comet")
        return None

    if not comet_model_path or not os.path.exists(comet_model_path):
        logger.warning("COMET checkpoint 不存在: %s，跳过 COMET 评估。", comet_model_path)
        return None

    try:
        model = load_from_checkpoint(comet_model_path)
        logger.info("COMET 模型加载成功: %s", comet_model_path)
        return model
    except Exception as e:
        logger.warning("COMET 模型加载失败: %s", e)
        import traceback
        traceback.print_exc()
        return None


def compute_comet_scores(
    model, srcs: list[str], hyps: list[str], refs: list[str], batch_size: int = 64
) -> list[Optional[float]]:
    if not srcs:
        return []

    items = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(srcs, hyps, refs)]

    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    scores: list[Optional[float]] = []
    ranges = list(range(0, len(items), batch_size))
    iterator = tqdm(ranges, desc="COMET scoring", unit="batch") if has_tqdm else ranges

    for start in iterator:
        end = min(start + batch_size, len(items))
        chunk = items[start:end]
        try:
            out = model.predict(chunk, batch_size=len(chunk))
            if hasattr(out, "scores"):
                batch_scores = out.scores
            elif isinstance(out, dict) and "scores" in out:
                batch_scores = out["scores"]
            else:
                batch_scores = [None] * len(chunk)

            scores.extend(batch_scores[:len(chunk)])
        except Exception as e:
            logger.warning("COMET batch %d-%d 失败: %s", start, end - 1, e)
            scores.extend([None] * len(chunk))

    return scores


def compute_comet_document(
    model, srcs: list[str], hyps: list[str], refs: list[str]
) -> Optional[float]:
    try:
        src_cat = "\n".join(srcs)
        mt_cat = "\n".join(hyps)
        ref_cat = "\n".join(refs)
        out = model.predict(
            [{"src": src_cat, "mt": mt_cat, "ref": ref_cat}], batch_size=1
        )
        if hasattr(out, "system_score"):
            return float(out.system_score)
        if hasattr(out, "scores") and out.scores:
            return float(out.scores[0])
        return None
    except Exception as e:
        logger.warning("Document-level COMET 失败: %s", e)
        return None


def main():
    parser = argparse.ArgumentParser(description="翻译质量评估（BLEU / spBLEU / chrF / chrF++ / COMET）")
    parser.add_argument("--translations", required=True,
                        help="翻译结果 JSONL（run_translate.py 输出）")
    parser.add_argument("--tgt_lang", required=True,
                        help="目标语言码（决定 BLEU tokenizer：zh→中文分词，其他→13a）")
    parser.add_argument("--metrics_output", required=True,
                        help="数据集级指标输出 JSON 路径")
    parser.add_argument("--per_sample_output", default=None,
                        help="逐句评估结果输出 JSON 路径（可选）")
    parser.add_argument("--comet_model", default=None,
                        help="COMET checkpoint 本地路径（wmt22-comet-da）")
    parser.add_argument("--local_xlm_path", default=None,
                        help="xlm-roberta-large 本地目录（COMET 离线模式）")
    parser.add_argument("--comet_batch_size", type=int, default=64,
                        help="COMET 推理批大小")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="每 N 句打印一次翻译和评分日志")
    args = parser.parse_args()

    records = read_jsonl(args.translations)
    if not records:
        logger.error("翻译结果文件为空或不存在: %s", args.translations)
        sys.exit(1)
    logger.info("加载 %d 条翻译记录: %s", len(records), args.translations)

    try:
        import sacrebleu
        logger.info("sacrebleu 版本: %s", sacrebleu.__version__)
    except ImportError:
        logger.error("sacrebleu 未安装！请运行: pip install 'sacrebleu>=2.0.0'")
        sys.exit(1)

    comet_model_obj = load_comet_model(args.comet_model, args.local_xlm_path)
    use_comet = comet_model_obj is not None
    logger.info("COMET 可用: %s", use_comet)

    logger.info("开始逐句计算 BLEU / spBLEU / chrF / chrF++...")
    per_sample_results: list[dict] = []
    all_srcs, all_hyps, all_refs = [], [], []

    bleu_tok = _get_bleu_tokenizer(args.tgt_lang)
    logger.info("BLEU tokenizer: tgt_lang=%s → tokenize='%s'", args.tgt_lang, bleu_tok)

    for i, rec in enumerate(records):
        idx = rec.get("idx", i)
        src = rec.get("src", "") or ""
        hyp = rec.get("translation", "") or ""
        ref = rec.get("ref", "") or ""

        if not ref:
            logger.warning("第 %d 条记录缺少参考翻译(ref)，跳过。", idx)
            continue

        scores = compute_sentence_metrics(hyp, ref, src, args.tgt_lang)

        sample = {
            "idx": idx,
            "src": src,
            "translation": hyp,
            "ref": ref,
            **scores,
            "comet": None, 
        }
        per_sample_results.append(sample)
        all_srcs.append(src)
        all_hyps.append(hyp)
        all_refs.append(ref)

        if args.log_interval > 0 and (i + 1) % args.log_interval == 0:
            logger.info(
                "[%d/%d] %s\n"
                "  SRC: %s\n"
                "  REF: %s\n"
                "  HYP: %s\n"
                "  BLEU=%.2f spBLEU=%.2f chrF=%.2f chrF++=%.2f",
                i + 1, len(records),
                rec.get("lang_pair", args.tgt_lang),
                src[:120] + ("..." if len(src) > 120 else ""),
                ref[:120] + ("..." if len(ref) > 120 else ""),
                hyp[:120] + ("..." if len(hyp) > 120 else ""),
                scores["bleu"], scores["sp_bleu"], scores["chrf"], scores["chrfpp"],
            )

    logger.info("逐句计算完成: %d / %d 条有效（有参考翻译）。", len(per_sample_results), len(records))

    comet_doc_score = None
    if use_comet and all_srcs:
        logger.info("开始 COMET sentence-level 评分...")
        comet_scores = compute_comet_scores(
            comet_model_obj, all_srcs, all_hyps, all_refs,
            batch_size=args.comet_batch_size,
        )
        for j, cs in enumerate(comet_scores):
            if cs is not None:
                per_sample_results[j]["comet"] = round(float(cs) * 100.0, 4)

        logger.info("计算 document-level COMET...")
        comet_doc_score = compute_comet_document(
            comet_model_obj, all_srcs, all_hyps, all_refs
        )
        if comet_doc_score is not None:
            comet_doc_score = round(float(comet_doc_score) * 100.0, 2)

    corpus = compute_corpus_metrics(all_hyps, all_refs, args.tgt_lang)

    valid_comet = [s["comet"] for s in per_sample_results if s["comet"] is not None]
    comet_mean = round(sum(valid_comet) / len(valid_comet), 2) if valid_comet else None

    dataset_metrics = {
        "n_samples": len(per_sample_results),
        "BLEU": corpus["BLEU"],
        "spBLEU": corpus["spBLEU"],
        "chrF": corpus["chrF"],
        "chrF++": corpus["chrF++"],
        "COMET": comet_mean,
        "COMET_document": comet_doc_score,
    }

    write_json(args.metrics_output, dataset_metrics)
    logger.info("数据集级指标已保存: %s", args.metrics_output)
    logger.info("评估结果: %s", json.dumps(dataset_metrics, ensure_ascii=False))

    if args.per_sample_output:
        write_json(args.per_sample_output, per_sample_results)
        logger.info("逐句评估结果已保存: %s", args.per_sample_output)

    logger.info("=" * 60)
    logger.info("评估完成  样本数: %d", dataset_metrics["n_samples"])
    logger.info(
        "BLEU=%.2f  spBLEU=%.2f  chrF=%.2f  chrF++=%.2f  COMET=%s  COMET_doc=%s",
        dataset_metrics["BLEU"],
        dataset_metrics["spBLEU"],
        dataset_metrics["chrF"],
        dataset_metrics["chrF++"],
        f'{dataset_metrics["COMET"]:.2f}' if dataset_metrics["COMET"] is not None else "N/A",
        f'{dataset_metrics["COMET_document"]:.2f}' if dataset_metrics["COMET_document"] is not None else "N/A",
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
