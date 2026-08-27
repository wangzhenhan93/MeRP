# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
#
# Licensed under the Apache License, Version 2.0 (the "License"),
#

import os
import json
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union, Dict, List
import math
import numpy as np
import torch
from transformers import Seq2SeqTrainer, TrainerCallback
from typing_extensions import override

from ...multilingual_prompt.prompt_manager import PromptManager
from ...multilingual_prompt.prompt_injector import attach_prompt_to_inputs_embeds
from ...multilingual_prompt.prompt_trainer_adapter import PromptTrainerAdapter
from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler
from .save_prompt_callback import SavePromptCallback, EncoderCheckpointCallback
from datasets import Dataset, DatasetDict

if TYPE_CHECKING:
    from torch.utils.data import Dataset as TorchDataset
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput
    from ...hparams import FinetuningArguments

logger = logging.get_logger(__name__)


def _concat_prompt_by_position(
    input_embeds: torch.Tensor,
    prompt_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    labels: Optional[torch.Tensor],
    position: str,
):
    bsz = input_embeds.size(0)
    seq_len = input_embeds.size(1)
    p_len = prompt_embeds.size(1)
    device = input_embeds.device

    pos = str(position or "prefix").lower()
    if pos not in ("prefix", "suffix"):
        pos = "prefix"

    if pos == "suffix":
        new_inputs_embeds = torch.cat([input_embeds, prompt_embeds], dim=1)
        if attention_mask is not None:
            suffix_mask = torch.ones((bsz, p_len), dtype=attention_mask.dtype, device=device)
            new_attention_mask = torch.cat([attention_mask, suffix_mask], dim=1)
        else:
            new_attention_mask = torch.ones((bsz, seq_len + p_len), dtype=torch.long, device=device)
        if labels is not None:
            pad = torch.full((bsz, p_len), IGNORE_INDEX, device=device, dtype=labels.dtype)
            new_labels = torch.cat([labels, pad], dim=1)
        else:
            new_labels = None
        return new_inputs_embeds, new_attention_mask, new_labels

    
    new_inputs_embeds = torch.cat([prompt_embeds, input_embeds], dim=1)
    if attention_mask is not None:
        prefix_mask = torch.ones((bsz, p_len), dtype=attention_mask.dtype, device=device)
        new_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
    else:
        new_attention_mask = torch.ones((bsz, p_len + seq_len), dtype=torch.long, device=device)

    if labels is not None:
        pad = torch.full((bsz, p_len), IGNORE_INDEX, device=device, dtype=labels.dtype)
        new_labels = torch.cat([pad, labels], dim=1)
    else:
        new_labels = None

    return new_inputs_embeds, new_attention_mask, new_labels


class DataCollatorWithRaw:
    def __init__(self, base_collator=None):
        if base_collator is None:
            raise ValueError(
                "DataCollatorWithRaw requires a base_collator. "
                "SDA-RA does not support collator-less training."
            )
        self.base_collator = base_collator

    def __call__(self, features):
        batch = self.base_collator(features)
        if batch is None:
            raise ValueError("DataCollatorWithRaw: base_collator returned None")
        return batch


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
  
    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        gen_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: "PreTrainedTokenizer" = kwargs.get("tokenizer")
        self.data_args = kwargs.pop("data_args", None)

        super().__init__(**kwargs)
        self.base_model = self.model.module if hasattr(self.model, "module") else self.model
        self.finetuning_args = finetuning_args
        self.model_accepts_loss_kwargs = False

        
        ptype = str(getattr(finetuning_args, "prompt_encoder_type", "none") or "none").lower()
        self._prompt_encoder_enabled = ptype != "none"
        self._prompt_disabled = not self._prompt_encoder_enabled
        self._prompt_save_dir = getattr(finetuning_args, "prompt_save_dir", "prompts")
        self._save_prompts_enabled = bool(getattr(finetuning_args, "save_prompts", True))
        self._vector_prompt_position = str(getattr(finetuning_args, "vector_prompt_position", "prefix")).lower()
        self._prompt_lr = getattr(finetuning_args, "prompt_encoder_lr", getattr(self.args, "learning_rate", 1e-3))
        self._lang_pairs = getattr(finetuning_args, "lang_pairs", None)
        self._lang_loss_weights: Dict[str, float] = getattr(finetuning_args, "_parsed_lang_loss_weights", {})
        self._contrastive_loss_weight = float(getattr(finetuning_args, "sda_contrastive_loss_weight", 0.1))
        self._balance_loss_weight = float(getattr(finetuning_args, "sda_balance_loss_weight", 0.01))
        self._anchor_loss_enabled = bool(getattr(finetuning_args, "sda_enable_anchor_loss", False))
        self._anchor_loss_weight_start = float(getattr(finetuning_args, "sda_anchor_loss_weight_start", 0.30))
        self._anchor_loss_weight_end = float(getattr(finetuning_args, "sda_anchor_loss_weight_end", 0.05))
        self._log_interval_lang_loss_sums: Dict[str, float] = {}
        self._log_interval_lang_loss_counts: Dict[str, int] = {}
        self._log_interval_metric_sums: Dict[str, float] = {}
        self._log_interval_metric_counts: Dict[str, int] = {}
        self._latest_loss_snapshot: Dict[str, float] = {}
        self._latest_direction_losses: Dict[str, float] = {}
        self._train_metrics_jsonl = os.path.join(self.args.output_dir, "train_metrics.jsonl") if getattr(self.args, "output_dir", None) else None

        self._last_post_clip_grad_norm: Optional[float] = None
        ft_gclip = getattr(finetuning_args, "finetuning_max_grad_norm", None)
        if ft_gclip is not None:
            try:
                self.args.max_grad_norm = float(ft_gclip) if float(ft_gclip) > 0 else 0.0
            except Exception:
                pass

        try:
            tokenizer = self.processing_class
        except Exception:
            tokenizer = kwargs.get("tokenizer", None)

        if self._prompt_disabled:
            logger.info_rank0("[Prompt] Disabled (prompt_encoder_type=none).")
            self.prompt_manager = None
            self.prompt_adapter = None
        else:
            if tokenizer is None:
                raise RuntimeError("[Prompt] Tokenizer not found while SDA-RA is enabled.")
            self.prompt_manager = PromptManager(
                self.model, tokenizer, save_dir=self._prompt_save_dir, args=finetuning_args
            )
            self.prompt_adapter = PromptTrainerAdapter(self.prompt_manager)
            logger.info_rank0(f"[Prompt] PromptManager initialized. save_dir={self._prompt_save_dir}")

            if self._lang_pairs:
                self.prompt_manager.ensure_prompts(self._lang_pairs)
                logger.info_rank0(f"[Prompt] ensure_prompts done for {self._lang_pairs}")

        if self.prompt_manager is not None and self._save_prompts_enabled:
            self.add_callback(SavePromptCallback(self.prompt_manager, trainer=self, save_mode="active_only"))
            _encoder_save_steps = int(getattr(finetuning_args, "encoder_save_steps", 0) or 0)
            if _encoder_save_steps > 0:
                self.add_callback(EncoderCheckpointCallback(
                    prompt_manager=self.prompt_manager,
                    encoder_save_steps=_encoder_save_steps,
                    prompt_save_dir=self._prompt_save_dir,
                ))

        base_coll = getattr(self, "data_collator", None)
        self.data_collator = DataCollatorWithRaw(base_collator=base_coll)

        if gen_kwargs is not None:
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if getattr(finetuning_args, "use_badam", False):
            from badam import BAdamCallback, clip_grad_norm_old_version
            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        if getattr(finetuning_args, "use_dft_loss", False):
            from ..trainer_utils import dft_loss_func
            self.compute_loss_func = dft_loss_func

        self.add_callback(_TrainEndCallback(trainer_ref=self))


    def _sync_prompt_encoders_to_model(self) -> None:
        if self.prompt_manager is None:
            return
        try:
            ref_weight = self.base_model.get_input_embeddings().weight
            target_device = ref_weight.device
            target_dtype = ref_weight.dtype
        except Exception:
            return

        enc = getattr(self.model, self.prompt_manager.SHARED_NAME, None)
        if enc is None or not isinstance(enc, torch.nn.Module):
            return

        enc.to(device=target_device)
        for name, param in enc.named_parameters():
            if name == "output_scale":
                param.data = param.data.to(device=target_device)  # keep fp32
            else:
                if param.dtype != target_dtype:
                    param.data = param.data.to(dtype=target_dtype)


    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is not None:
            return self.optimizer

        if self._prompt_disabled or self.prompt_manager is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
            return super().create_optimizer()

        if self._lang_pairs:
            self.prompt_manager.ensure_prompts(self._lang_pairs)

        self._sync_prompt_encoders_to_model()

        for p in self.model.parameters():
            p.requires_grad = False

        params = self.prompt_adapter.activate_all(self.model, freeze_model=True)

        if len(params) == 0:
            raise RuntimeError(
                "[create_optimizer] No SDA-RA parameters found. "
                "Check PromptManager initialization."
            )

        lr = self._prompt_lr if self._prompt_lr is not None else getattr(self.args, "learning_rate", 1e-3)
        self.optimizer = torch.optim.AdamW(params, lr=lr)

        total_trainable = sum(p.numel() for p in params)
        logger.info_rank0(
            f"[create_optimizer] SDA-RA parameters: {len(params)} tensors, "
            f"{total_trainable} total scalars ({total_trainable/1e6:.3f}M), lr={lr}"
        )

        if self.is_world_process_zero():
            param_ids = {id(p) for p in params}
            for name, p in self.model.named_parameters():
                if id(p) in param_ids:
                    logger.info_rank0(f"  [opt] {name:60s} {str(tuple(p.shape)):>20s} {p.numel()}")

        return self.optimizer

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if getattr(self.finetuning_args, "disable_shuffling", False):
            return torch.utils.data.SequentialSampler(self.train_dataset)
        return super()._get_train_sampler(*args, **kwargs)


    def _resolve_lang_pair_id(self, lang_pair: str, device: torch.device) -> Optional[torch.Tensor]:
        has_local_mapping = False
        try:
            if self.prompt_manager is not None:
                has_local_mapping = bool(getattr(self.prompt_manager, "lang_pair_to_id", None))
                _val = self.prompt_manager.resolve_lang_pair_id(str(lang_pair))
                if _val is not None:
                    return torch.tensor([int(_val)], device=device)
        except Exception as e:
            logger.warning_rank0(f"[_resolve_lang_pair_id] prompt_manager map lookup failed for '{lang_pair}': {e}")

        if has_local_mapping:
            return None

        try:
            _candidates = [
                os.path.join(os.getcwd(), "data", "lang_pair_map.json"),
                os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", "..", "..", "..", "data", "lang_pair_map.json"),
                getattr(getattr(self, "data_args", None), "lang_map_path", "") or "",
            ]
            for _cand in _candidates:
                if _cand and os.path.isfile(_cand):
                    with open(_cand, "r") as _f:
                        _lm = json.load(_f)
                    _val = _lm.get(str(lang_pair))
                    if _val is not None:
                        return torch.tensor([int(_val)], device=device)
                    break
        except Exception as e:
            logger.warning_rank0(f"[_resolve_lang_pair_id] failed for '{lang_pair}': {e}")
        return None

    def _resolve_lang_pair_ids(
        self,
        lang_pairs: Any,
        device: torch.device,
        fallback_ids: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if lang_pairs is None:
            if fallback_ids is None:
                return None
            return fallback_ids.to(device)

        if isinstance(lang_pairs, torch.Tensor):
            lang_pairs = lang_pairs.tolist()
        elif isinstance(lang_pairs, (str, bytes)):
            lang_pairs = [lang_pairs]
        else:
            lang_pairs = list(lang_pairs)

        resolved_ids: list[int] = []
        unresolved_pairs: list[str] = []
        for lp in lang_pairs:
            if isinstance(lp, bytes):
                lp = lp.decode("utf-8", errors="ignore")
            lp_str = str(lp)
            lp_id = self._resolve_lang_pair_id(lp_str, device)
            if lp_id is None:
                unresolved_pairs.append(lp_str)
            else:
                resolved_ids.append(int(lp_id.item()))

        if not unresolved_pairs:
            return torch.tensor(resolved_ids, dtype=torch.long, device=device)

        if fallback_ids is not None:
            fallback_ids = fallback_ids.to(device=device, dtype=torch.long).view(-1)
            if fallback_ids.numel() == len(lang_pairs):
                logger.warning_rank0(
                    f"[_resolve_lang_pair_ids] fallback to batch lang_pair_id for unresolved pairs: {unresolved_pairs}"
                )
                return fallback_ids

        raise ValueError(
            "[SDA-RA] 无法为当前 batch 解析紧凑 lang_pair_id。"
            f" unresolved={unresolved_pairs}."
            " 请检查 YAML 中的 lang_pairs 是否覆盖这些方向，"
            "并确认当前方向编号与训练配置一致。"
        )

    @override
    def compute_loss(self, model, inputs, *args, **kwargs):
        if self.prompt_manager is None or self._prompt_disabled:
            return super().compute_loss(model, inputs, *args, **kwargs)

        lang_pairs = inputs.pop("lang_pair", None)
        if lang_pairs is None:
            raise ValueError("[compute_loss] lang_pair missing from inputs.")

        if isinstance(lang_pairs, torch.Tensor):
            lang_pairs = lang_pairs.tolist()

        device = next(model.parameters()).device
        gstep = int(getattr(self.state, "global_step", 0) or 0)

        input_ids = inputs.pop("input_ids").to(device)
        attention_mask = inputs.pop("attention_mask", None)
        labels = inputs.pop("labels", None)
        encoder_ids = inputs.pop("encoder_input_ids", None)
        encoder_mask = inputs.pop("encoder_attention_mask", None)
        lang_pair_id = inputs.pop("lang_pair_id", None)

        for k in ("raw_input_ids", "raw_attention_mask", "raw_inputs_embeds", "src_text"):
            inputs.pop(k, None)

        if encoder_ids is None or encoder_ids.numel() == 0:
            raise ValueError("[compute_loss] encoder_input_ids missing.")

        encoder_ids = encoder_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if labels is not None:
            labels = labels.to(device)
        if encoder_mask is not None:
            encoder_mask = encoder_mask.to(device)
        else:
            pad_id = getattr(self.processing_class, "pad_token_id", None)
            if pad_id is None:
                encoder_mask = torch.ones(encoder_ids.shape[:2], dtype=torch.long, device=device)
            else:
                encoder_mask = (encoder_ids != pad_id).long().to(device)
        lang_pair_id = self._resolve_lang_pair_ids(lang_pairs, device, lang_pair_id)

        bsz = input_ids.size(0)

        with torch.no_grad():
            inputs_embeds = self.base_model.get_input_embeddings()(input_ids)
            enc_embeds = self.base_model.get_input_embeddings()(encoder_ids)

        enc_mod = getattr(self.model, self.prompt_manager.SHARED_NAME)
        prompt_embeddings = enc_mod(
            enc_embeds,
            encoder_mask,
            lang_pair_id=lang_pair_id,
        )

        new_inputs_embeds, new_attention_mask, new_labels = _concat_prompt_by_position(
            input_embeds=inputs_embeds,
            prompt_embeds=prompt_embeddings,
            attention_mask=attention_mask,
            labels=labels,
            position=self._vector_prompt_position,
        )

        has_custom_weights = bool(self._lang_loss_weights) and any(
            self._lang_loss_weights.get(lp, 1.0) != 1.0 for lp in lang_pairs
        )

        if not has_custom_weights:
            outputs = model(
                inputs_embeds=new_inputs_embeds,
                attention_mask=new_attention_mask,
                labels=new_labels,
            )
            main_loss = outputs.loss
        else:
            outputs = model(
                inputs_embeds=new_inputs_embeds,
                attention_mask=new_attention_mask,
            )
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = new_labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=IGNORE_INDEX)
            per_token = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            ).view(bsz, -1)
            valid = (shift_labels != IGNORE_INDEX).float()
            per_sample = (per_token * valid).sum(1) / valid.sum(1).clamp(min=1)

            w = torch.tensor(
                [self._lang_loss_weights.get(lp, 1.0) for lp in lang_pairs],
                dtype=torch.float32, device=device,
            )
            main_loss = (per_sample * w).sum() / w.sum()

        loss_metrics: Dict[str, float] = {
            "loss_main": float(main_loss.detach().item()),
        }

        try:
            loss_val = float(main_loss.detach().item())
            grouped: Dict[str, List[int]] = {}
            for i, lp in enumerate(lang_pairs):
                grouped.setdefault(lp, []).append(i)
            for lp, indices in grouped.items():
                n = len(indices)
                self._log_interval_lang_loss_sums[lp] = (
                    self._log_interval_lang_loss_sums.get(lp, 0.0) + loss_val * n
                )
                self._log_interval_lang_loss_counts[lp] = (
                    self._log_interval_lang_loss_counts.get(lp, 0) + n
                )
        except Exception:
            pass

        enc = getattr(self.model, self.prompt_manager.SHARED_NAME, None)
        aux_loss = torch.tensor(0.0, device=device)

        if enc is not None and self._contrastive_loss_weight > 0:
            contrastive = enc.compute_contrastive_loss()
            contrastive_weighted = self._contrastive_loss_weight * contrastive
            aux_loss = aux_loss + contrastive_weighted
            loss_metrics["loss_contrastive_raw"] = float(contrastive.detach().item())
            loss_metrics["loss_contrastive"] = float(contrastive_weighted.detach().item())
        else:
            loss_metrics["loss_contrastive_raw"] = 0.0
            loss_metrics["loss_contrastive"] = 0.0

        if enc is not None and bool(getattr(enc, "enable_balance_loss", False)) and self._balance_loss_weight > 0:
            balance = enc.compute_balance_loss()
            balance_weighted = self._balance_loss_weight * balance
            aux_loss = aux_loss + balance_weighted
            loss_metrics["loss_balance_raw"] = float(balance.detach().item())
            loss_metrics["loss_balance"] = float(balance_weighted.detach().item())
        else:
            loss_metrics["loss_balance_raw"] = 0.0
            loss_metrics["loss_balance"] = 0.0

        if enc is not None and self._anchor_loss_enabled:
            anchor_weight = self._get_current_anchor_loss_weight()
            loss_metrics["loss_anchor_weight"] = float(anchor_weight)
            if anchor_weight > 0:
                anchor_loss = enc.compute_anchor_loss()
                anchor_weighted = anchor_weight * anchor_loss
                aux_loss = aux_loss + anchor_weighted
                loss_metrics["loss_anchor_raw"] = float(anchor_loss.detach().item())
                loss_metrics["loss_anchor"] = float(anchor_weighted.detach().item())
            else:
                loss_metrics["loss_anchor_raw"] = 0.0
                loss_metrics["loss_anchor"] = 0.0
        else:
            loss_metrics["loss_anchor_weight"] = 0.0
            loss_metrics["loss_anchor_raw"] = 0.0
            loss_metrics["loss_anchor"] = 0.0

        loss_metrics["loss_aux"] = float(aux_loss.detach().item())

        total_loss = main_loss + aux_loss
        loss_metrics["loss_total"] = float(total_loss.detach().item())

        try:
            for metric_name, metric_val in loss_metrics.items():
                self._log_interval_metric_sums[metric_name] = self._log_interval_metric_sums.get(metric_name, 0.0) + float(metric_val)
                self._log_interval_metric_counts[metric_name] = self._log_interval_metric_counts.get(metric_name, 0) + 1
        except Exception:
            pass

        return total_loss

    def _get_current_anchor_loss_weight(self) -> float:
        if not self._anchor_loss_enabled:
            return 0.0

        start = float(self._anchor_loss_weight_start)
        end = float(self._anchor_loss_weight_end)
        max_steps = int(getattr(self.state, "max_steps", 0) or getattr(self.args, "max_steps", 0) or 0)
        if max_steps <= 0:
            return start

        step = float(getattr(self.state, "global_step", 0) or 0)
        progress = min(max(step / float(max_steps), 0.0), 1.0)
        return end + (start - end) * (1.0 - progress)


    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        if self.args.predict_with_generate:
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        lp = None
        if "lang_pair" in inputs:
            lp = inputs.get("lang_pair")
            if isinstance(lp, (list, tuple)):
                lp = lp[0]
            if isinstance(lp, torch.Tensor):
                lp = str(lp.item()) if lp.dim() == 0 else str(lp[0].item())

        if lp is None:
            lp = getattr(self.finetuning_args, "current_lang_pair", None)
        if lp is None:
            lp_list = getattr(self.finetuning_args, "lang_pairs", None)
            if lp_list:
                lp = lp_list[0]

        prompt_len = 0
        if lp is not None and self.prompt_manager is not None and not self._prompt_disabled:
            input_ids = inputs.pop("input_ids")
            attention_mask = inputs.pop("attention_mask", None)
            device = next(model.parameters()).device

            inputs_embeds = self.base_model.get_input_embeddings()(input_ids.to(device))

            encoder_ids = inputs.pop("encoder_input_ids", None)
            encoder_mask = inputs.pop("encoder_attention_mask", None)
            lang_pair_id = inputs.pop("lang_pair_id", None)

            if encoder_ids is None or encoder_ids.numel() == 0:
                raise ValueError(f"[prediction_step] encoder_input_ids missing for '{lp}'.")

            encoder_ids = encoder_ids.to(device)
            enc_embeds = self.base_model.get_input_embeddings()(encoder_ids)
            if encoder_mask is not None:
                encoder_mask = encoder_mask.to(device)
            else:
                pad_id = getattr(self.processing_class, "pad_token_id", None)
                if pad_id is None:
                    encoder_mask = torch.ones(encoder_ids.shape[:2], dtype=torch.long, device=device)
                else:
                    encoder_mask = (encoder_ids != pad_id).long()

            if lp is not None:
                lang_pair_id = self._resolve_lang_pair_ids([lp], device, lang_pair_id)

            enc_mod = self.prompt_manager.get_prompt(lp)
            prompt_param = enc_mod(
                enc_embeds,
                encoder_mask,
                lang_pair_id=lang_pair_id,
            )

            if prompt_param.dim() != 3:
                raise RuntimeError(
                    f"[prediction_step] Encoder returned {prompt_param.shape}, expected (B, P, D)"
                )

            if attention_mask is not None and attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)

            new_inputs_embeds, new_attention_mask, _ = _concat_prompt_by_position(
                input_embeds=inputs_embeds,
                prompt_embeds=prompt_param,
                attention_mask=attention_mask,
                labels=None,
                position=self._vector_prompt_position,
            )

            prompt_len = prompt_param.size(1)
            inputs["inputs_embeds"] = new_inputs_embeds
            inputs["attention_mask"] = new_attention_mask

        for key in ("lang_pair", "src_text", "raw_input_ids", "raw_attention_mask",
                     "raw_inputs_embeds", "encoder_input_ids", "encoder_attention_mask",
                     "lang_pair_id"):
            inputs.pop(key, None)

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )

        return loss, generated_tokens, labels


    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):
                preds[i] = np.concatenate((preds[i][pad_len[0]:], preds[i][:pad_len[0]]), axis=-1)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")


    @override
    def optimizer_step(self, epoch: Optional[int] = None, batch_idx: Optional[int] = None,
                       optimizer: Optional[torch.optim.Optimizer] = None, optimizer_closure=None, **kwargs):
        opt = optimizer if optimizer is not None else self.optimizer
        if opt is None:
            return super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure, **kwargs)

        max_norm = float(getattr(self.args, "max_grad_norm", 0.0) or 0.0)
        params_to_clip = [p for p in self.model.parameters() if p.requires_grad and p.grad is not None]

        post_norm = None
        if len(params_to_clip) > 0 and max_norm > 0.0:
            try:
                if hasattr(self, "accelerator") and self.accelerator is not None:
                    self.accelerator.clip_grad_norm_(params_to_clip, max_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm)
            except Exception:
                torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm)

            total = sum(float(p.grad.detach().norm(2).item()) ** 2 for p in params_to_clip)
            post_norm = total ** 0.5
            self._last_post_clip_grad_norm = post_norm

        orig_max = getattr(self.args, "max_grad_norm", None)
        try:
            self.args.max_grad_norm = 0.0
            return super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure, **kwargs)
        finally:
            self.args.max_grad_norm = orig_max


    @override
    def log(self, logs, start_time=None):
        if isinstance(logs, dict):
            post = getattr(self, "_last_post_clip_grad_norm", None)
            if post is not None and "grad_norm" in logs:
                logs["grad_norm_pre_clip"] = logs.get("grad_norm")
                logs["grad_norm"] = float(post)
                self._last_post_clip_grad_norm = None

            if self._log_interval_lang_loss_counts:
                for _lp in sorted(self._log_interval_lang_loss_counts.keys()):
                    _s = self._log_interval_lang_loss_sums.get(_lp, 0.0)
                    _c = self._log_interval_lang_loss_counts.get(_lp, 0)
                    if _c > 0:
                        logs[f"loss_{_lp}"] = round(_s / _c, 6)
                        self._latest_direction_losses[_lp] = round(_s / _c, 6)
                self._log_interval_lang_loss_sums = {}
                self._log_interval_lang_loss_counts = {}

            if self._log_interval_metric_counts:
                for _name in sorted(self._log_interval_metric_counts.keys()):
                    _s = self._log_interval_metric_sums.get(_name, 0.0)
                    _c = self._log_interval_metric_counts.get(_name, 0)
                    if _c > 0:
                        logs[_name] = round(_s / _c, 6)
                        self._latest_loss_snapshot[_name] = round(_s / _c, 6)
                self._log_interval_metric_sums = {}
                self._log_interval_metric_counts = {}

            if self.is_world_process_zero() and self._train_metrics_jsonl:
                try:
                    os.makedirs(os.path.dirname(self._train_metrics_jsonl), exist_ok=True)
                    serializable_logs: Dict[str, Any] = {}
                    for _k, _v in logs.items():
                        if isinstance(_v, (str, bool, int, float)):
                            serializable_logs[_k] = _v
                        elif isinstance(_v, np.generic):
                            serializable_logs[_k] = _v.item()
                        elif torch.is_tensor(_v) and _v.numel() == 1:
                            serializable_logs[_k] = _v.item()
                    if "step" not in serializable_logs:
                        serializable_logs["step"] = int(getattr(self.state, "global_step", 0) or 0)
                    with open(self._train_metrics_jsonl, "a", encoding="utf-8") as f:
                        f.write(json.dumps(serializable_logs, ensure_ascii=False) + "\n")
                except Exception as e:
                    logger.warning_rank0(f"[log] failed to append train_metrics.jsonl: {e}")

        return super().log(logs, start_time=start_time)


    def _run_train_end_logic(self, args, state, control, **kwargs):
        try:
            if (
                self.prompt_manager is not None
                and not self._prompt_disabled
                and self._save_prompts_enabled
                and self.is_world_process_zero()
            ):
                saved = self.prompt_manager.save_prompts(self._prompt_save_dir)
                logger.info_rank0(f"[on_train_end] Saved SDA-RA encoder: {saved}")

                enc = getattr(self.model, self.prompt_manager.SHARED_NAME, None)
                if enc is not None and hasattr(enc, "get_routing_info"):
                    info = enc.get_routing_info()
                    summary_path = os.path.join(args.output_dir, "routing_final.json")
                    serializable = {}
                    for k, v in info.items():
                        if isinstance(v, torch.Tensor):
                            serializable[k] = v.tolist()
                        else:
                            serializable[k] = v
                    serializable["training_loss_summary"] = {
                        "latest_loss_snapshot": dict(self._latest_loss_snapshot),
                        "latest_direction_losses": dict(self._latest_direction_losses),
                        "metrics_jsonl": os.path.basename(self._train_metrics_jsonl) if self._train_metrics_jsonl else None,
                    }
                    with open(summary_path, "w", encoding="utf-8") as f:
                        json.dump(serializable, f, ensure_ascii=False, indent=2)
                    logger.info_rank0(f"[on_train_end] Saved routing_final.json -> {summary_path}")

                final_loss_path = os.path.join(args.output_dir, "train_loss_final.json")
                final_loss_payload = {
                    "latest_loss_snapshot": dict(self._latest_loss_snapshot),
                    "latest_direction_losses": dict(self._latest_direction_losses),
                    "global_step": int(getattr(state, "global_step", 0) or 0),
                    "epoch": float(getattr(state, "epoch", 0.0) or 0.0),
                    "metrics_jsonl": os.path.basename(self._train_metrics_jsonl) if self._train_metrics_jsonl else None,
                }
                with open(final_loss_path, "w", encoding="utf-8") as f:
                    json.dump(final_loss_payload, f, ensure_ascii=False, indent=2)
                logger.info_rank0(f"[on_train_end] Saved train_loss_final.json -> {final_loss_path}")
        except Exception as e:
            logger.warning_rank0(f"[on_train_end] failed: {e}")


class _TrainEndCallback(TrainerCallback):
    def __init__(self, trainer_ref):
        self._trainer_ref = trainer_ref

    def on_train_end(self, args, state, control, **kwargs):
        try:
            self._trainer_ref._run_train_end_logic(args, state, control, **kwargs)
        except Exception as e:
            logger.warning_rank0(f"[_TrainEndCallback] on_train_end failed: {e}")
        return control
