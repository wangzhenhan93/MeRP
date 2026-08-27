# src/llamafactory/train/sft/supervised.py
# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional
import json
import os

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from .processor_utils import DatasetProcessor, greedy_knapsack, infer_seqlen

from ..parser import normalize_lang_pair  

if TYPE_CHECKING:
    from ..mm_plugin import AudioInput, ImageInput, VideoInput

logger = logging.get_logger(__name__)


@dataclass
class SupervisedDatasetProcessor(DatasetProcessor):
    def _encode_data_example(
        self,
        prompt: list[dict[str, str]],
        response: list[dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        lang_pair: Optional[str],  
    ) -> tuple[list[int], list[int]]:
        messages = self.template.mm_plugin.process_messages(prompt + response, images, videos, audios, self.processor)
        input_ids_partial, labels_partial = self.template.mm_plugin.process_token_ids(
            [], [], images, videos, audios, self.tokenizer, self.processor
        )
        encoded_pairs = self.template.encode_multiturn(self.tokenizer, messages, system, tools)

        input_ids: list[int] = []
        labels: list[int] = []

        total_length = 1 if self.template.efficient_eos else 0

        if self.data_args.mask_history:
            encoded_pairs = encoded_pairs[::-1]  
        for turn_idx, (source_ids, target_ids) in enumerate(encoded_pairs):
            if total_length >= self.data_args.cutoff_len:
                break

            source_len, target_len = infer_seqlen(
                len(source_ids), len(target_ids), self.data_args.cutoff_len - total_length
            )
            source_ids = source_ids[:source_len]
            target_ids = target_ids[:target_len]
            total_length += source_len + target_len

            if self.data_args.train_on_prompt:
                source_label = source_ids
            elif self.template.efficient_eos and turn_idx != 0:
                source_label = [self.tokenizer.eos_token_id] + [IGNORE_INDEX] * (source_len - 1)
            else:
                source_label = [IGNORE_INDEX] * source_len

            if self.data_args.mask_history and turn_idx != 0:
                target_label = [IGNORE_INDEX] * target_len
            else:
                target_label = target_ids

            if self.data_args.mask_history:
                input_ids = source_ids + target_ids + input_ids
                labels = source_label + target_label + labels
            else:
                input_ids += source_ids + target_ids
                labels += source_label + target_label
        if self.template.efficient_eos:
            input_ids += [self.tokenizer.eos_token_id]
            labels += [self.tokenizer.eos_token_id]

        return input_ids, labels

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        lang_candidates = ("_lang_pair", "lang_pair", "langpair", "pair", "_lang_pair_raw")

        logger.info_rank0(f"preprocess_dataset: keys={list(examples.keys())}")
        for k in lang_candidates:
            if k in examples:
                sample = examples[k][:8] if isinstance(examples[k], (list, tuple)) else examples[k]
                logger.info_rank0(f"preprocess_dataset: found key {k}; sample={sample}")
        for k in ("_prompt", "prompt", "instruction"):
            if k in examples:
                s = examples[k][:3] if isinstance(examples[k], (list, tuple)) else examples[k]
                logger.info_rank0(f"preprocess_dataset: prompt key {k} sample={s}")

        model_inputs = defaultdict(list)
        lang_map = {}
        lang_map_path = getattr(self.data_args, "lang_map_path", "data/lang_pair_map.json")
        if lang_map_path and os.path.exists(lang_map_path):
            try:
                with open(lang_map_path, "r", encoding="utf-8") as f:
                    lang_map = json.load(f)
                lang_map = {str(k): int(v) for k, v in lang_map.items()}
            except Exception as e:
                logger.info_rank0(f"Failed to load lang_map: {e}")
                lang_map = {}

        n_examples = len(examples.get("_prompt", examples.get("instruction", [])))
        raw_langs: list[str] = []
        forced_lp = getattr(self.data_args, "current_forced_lang_pair", None)
        if not forced_lp:
            forced_lp = getattr(self.data_args, "target_lang_pair", None)
        if forced_lp:
            forced_lp = normalize_lang_pair(str(forced_lp))

        def _get_field_val(batch: dict, key: str, idx: int):
            val = batch.get(key)
            if isinstance(val, (list, tuple)):
                try:
                    return val[idx]
                except Exception:
                    return val
            else:
                return val

        for i in range(n_examples):
            if forced_lp:
                lp = forced_lp
            else:
                lp = None
                for cand in lang_candidates:
                    if cand in examples:
                        lp = _get_field_val(examples, cand, i)
                        break
                if lp is None:
                    lp = "unknown"
                lp = normalize_lang_pair(lp)
            raw_langs.append(lp)

        next_idx = max(lang_map.values()) + 1 if lang_map else 0
        seen_new = False
        for s in raw_langs:
            if s not in lang_map:
                lang_map[s] = next_idx
                next_idx += 1
                seen_new = True
        for i in range(n_examples):
            lang_s = raw_langs[i]
            try:
                input_ids, labels = self._encode_data_example(
                    prompt=examples["_prompt"][i],
                    response=examples["_response"][i],
                    system=examples.get("_system", [""] * n_examples)[i],
                    tools=examples.get("_tools", [""] * n_examples)[i],
                    images=examples.get("_images", [None] * n_examples)[i] or [],
                    videos=examples.get("_videos", [None] * n_examples)[i] or [],
                    audios=examples.get("_audios", [None] * n_examples)[i] or [],
                    lang_pair=lang_s,
                )
            except Exception as e:
                logger.warning_rank0(f"Dropped invalid example idx={i}: {e}")
                continue

            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)
            model_inputs["images"].append(examples.get("_images", [None] * n_examples)[i])
            model_inputs["videos"].append(examples.get("_videos", [None] * n_examples)[i])
            model_inputs["audios"].append(examples.get("_audios", [None] * n_examples)[i])

            try:
                prompt_msgs = examples["_prompt"][i]
                src_text = ""
                for msg in reversed(prompt_msgs):
                    if isinstance(msg, dict) and msg.get("role") == "user" and msg.get("content"):
                        src_text = str(msg["content"]).strip()
                        break
                model_inputs["src_text"].append(src_text)
            except Exception:
                model_inputs["src_text"].append("")

        model_inputs["lang_pair"] = raw_langs
        if seen_new and lang_map_path:
            try:
                os.makedirs(os.path.dirname(lang_map_path), exist_ok=True)
                with open(lang_map_path, "w", encoding="utf-8") as f:
                    json.dump(lang_map, f, ensure_ascii=False, indent=2)
                logger.info_rank0(f"Saved lang_map to {lang_map_path}")
            except Exception as e:
                logger.info_rank0(f"Failed to save lang_map: {e}")

        try:
            unique_langs = sorted(set(raw_langs))
            logger.info_rank0(f"[DEBUG] preprocess raw_langs (first 16) = {raw_langs[:16]}")
            logger.info_rank0(f"[DEBUG] model_inputs['lang_pair'] (first 16) = {model_inputs.get('lang_pair', [])[:16]}")
            logger.info_rank0(f"[DEBUG] unique lang pairs in this batch = {unique_langs}")
        except Exception as e:
            logger.info_rank0(f"[DEBUG] preprocess debug failed: {e}")

        return model_inputs

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        valid_labels = [x for x in example["labels"] if x != IGNORE_INDEX]
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("label_ids:\n{}".format(example["labels"]))
        print(f"labels:\n{self.tokenizer.decode(valid_labels, skip_special_tokens=False)}")
        print(f"lang_pair: {example.get('lang_pair', 'unknown')}")


@dataclass
class PackedSupervisedDatasetProcessor(SupervisedDatasetProcessor):
    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        if "lang_pair" in examples and "_lang_pair" not in examples:
            examples["_lang_pair"] = examples.pop("lang_pair")

        logger.info_rank0(f"preprocess_dataset (packed): keys={list(examples.keys())}")
        for k in ("_lang_pair", "lang_pair", "langpair", "pair", "_lang_pair_raw"):
            if k in examples:
                sample = examples[k][:8] if isinstance(examples[k], (list, tuple)) else examples[k]
                logger.info_rank0(f"preprocess_dataset: found key {k}; sample={sample}")
        for k in ("_prompt", "prompt", "instruction"):
            if k in examples:
                s = examples[k][:3] if isinstance(examples[k], (list, tuple)) else examples[k]
                logger.info_rank0(f"preprocess_dataset: prompt key {k} sample={s}")
        return super().preprocess_dataset(examples)
