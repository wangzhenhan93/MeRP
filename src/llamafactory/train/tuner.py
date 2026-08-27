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

import os
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.distributed as dist
from transformers import EarlyStoppingCallback, PreTrainedModel

from ..data import get_template_and_fix_tokenizer
from ..extras import logging
from ..extras.constants import V_HEAD_SAFE_WEIGHTS_NAME, V_HEAD_WEIGHTS_NAME
from ..extras.misc import infer_optim_dtype
from ..extras.packages import is_ray_available
from ..hparams import get_infer_args, get_train_args, read_args
from ..model import load_model, load_tokenizer

from .callbacks import LogCallback, PissaConvertCallback, ReporterCallback
from .dpo import run_dpo
from .kto import run_kto
from .ppo import run_ppo
from .pt import run_pt
from .rm import run_rm
from .sft import run_sft

if is_ray_available():
    import ray

if TYPE_CHECKING:
    from transformers import TrainerCallback

logger = logging.get_logger(__name__)



def _training_function(config: dict[str, Any]) -> None:
    args = config.get("args")
    callbacks: list[Any] = config.get("callbacks")

    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)

    callbacks.append(LogCallback())

    if finetuning_args.pissa_convert:
        callbacks.append(PissaConvertCallback())

    if finetuning_args.early_stopping_steps is not None:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=finetuning_args.early_stopping_steps
            )
        )

    callbacks.append(
        ReporterCallback(
            model_args, data_args, finetuning_args, generating_args
        )
    )

    stage = finetuning_args.stage

    if stage == "pt":
        run_pt(model_args, data_args, training_args, finetuning_args, callbacks)

    elif stage == "sft":
        run_sft(
            model_args,
            data_args,
            training_args,
            finetuning_args,
            generating_args,
            callbacks,
        )

    elif stage == "rm":
        run_rm(model_args, data_args, training_args, finetuning_args, callbacks)

    elif stage == "ppo":
        run_ppo(
            model_args,
            data_args,
            training_args,
            finetuning_args,
            generating_args,
            callbacks,
        )

    elif stage == "dpo":
        run_dpo(model_args, data_args, training_args, finetuning_args, callbacks)

    elif stage == "kto":
        run_kto(model_args, data_args, training_args, finetuning_args, callbacks)

    else:
        raise ValueError(f"Unknown training stage: {stage}")

    if is_ray_available() and ray.is_initialized():
        return

    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception as e:
        logger.warning(f"Failed to destroy process group: {e}.")

def run_exp(
    args: Optional[dict[str, Any]] = None,
    callbacks: Optional[list["TrainerCallback"]] = None,
) -> None:
    """
    Entry point for training / finetuning.

    Assumptions:
      - mixed language-pair datasets
      - sample-level lang_pair routing
      - encoder selection handled in trainer / prompt_manager
      - no sequential (per-lang) training
    """

    args = read_args(args)
    if "-h" in args or "--help" in args:
        get_train_args(args)

    callbacks = callbacks or []
    _training_function(config={"args": args, "callbacks": callbacks})


def export_model(args: Optional[dict[str, Any]] = None) -> None:
    model_args, data_args, finetuning_args, _ = get_infer_args(args)

    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    processor = tokenizer_module["processor"]

    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    model = load_model(tokenizer, model_args, finetuning_args)

    if not isinstance(model, PreTrainedModel):
        raise ValueError("The model is not a PreTrainedModel, export aborted.")

    if getattr(model, "quantization_method", None) is not None:
        setattr(model.config, "torch_dtype", torch.float16)
    else:
        if model_args.infer_dtype == "auto":
            output_dtype = getattr(model.config, "torch_dtype", torch.float32)
            if output_dtype == torch.float32:
                output_dtype = infer_optim_dtype(torch.bfloat16)
        else:
            output_dtype = getattr(torch, model_args.infer_dtype)

        setattr(model.config, "torch_dtype", output_dtype)
        model = model.to(output_dtype)
        logger.info_rank0(f"Convert model dtype to: {output_dtype}.")

    model.save_pretrained(
        save_directory=model_args.export_dir,
        max_shard_size=f"{model_args.export_size}GB",
        safe_serialization=(not model_args.export_legacy_format),
    )

    if finetuning_args.stage == "rm":
        base_path = (
            model_args.adapter_name_or_path[-1]
            if model_args.adapter_name_or_path
            else model_args.model_name_or_path
        )

        if os.path.exists(os.path.join(base_path, V_HEAD_SAFE_WEIGHTS_NAME)):
            torch.save(
                torch.load(os.path.join(base_path, V_HEAD_SAFE_WEIGHTS_NAME)),
                os.path.join(model_args.export_dir, V_HEAD_SAFE_WEIGHTS_NAME),
            )
        elif os.path.exists(os.path.join(base_path, V_HEAD_WEIGHTS_NAME)):
            torch.save(
                torch.load(os.path.join(base_path, V_HEAD_WEIGHTS_NAME)),
                os.path.join(model_args.export_dir, V_HEAD_WEIGHTS_NAME),
            )

    try:
        tokenizer.padding_side = "left"
        tokenizer.init_kwargs["padding_side"] = "left"
        tokenizer.save_pretrained(model_args.export_dir)

        if processor is not None:
            processor.save_pretrained(model_args.export_dir)

    except Exception as e:
        logger.warning_rank0(f"Cannot save tokenizer/processor: {e}.")

    with open(os.path.join(model_args.export_dir, "Modelfile"), "w", encoding="utf-8") as f:
        f.write(template.get_ollama_modelfile(tokenizer))
