# src/llamafactory/train/sft/save_prompt_callback.py
import os
import logging
from transformers import TrainerCallback
from typing import Optional

logger = logging.getLogger(__name__)


class SavePromptCallback(TrainerCallback):

    def __init__(self, prompt_manager, trainer=None, save_mode: str = "active_only"):
        self.prompt_manager = prompt_manager
        self.trainer_ref = trainer
        self.save_mode = save_mode  

    def _is_rank0(self, trainer) -> bool:
        try:
            return trainer.is_world_process_zero()
        except Exception:
            return True

    def on_save(self, args, state, control, **kwargs):
        trainer = kwargs.get("trainer", self.trainer_ref)
        if trainer is None:
            return

        if not self._is_rank0(trainer):
            return

        prompt_disabled = getattr(trainer, "_prompt_disabled", False)
        save_flag = getattr(trainer.finetuning_args, "save_prompts", True) if hasattr(trainer, "finetuning_args") else True
        if prompt_disabled or not save_flag:
            return

        save_dir = getattr(self.prompt_manager, "save_dir", None) or os.path.join(args.output_dir or ".", "prompts")

        try:
            self.prompt_manager.save_prompts(save_dir)
            logger.info(f"[SavePromptCallback] saved SDA-RA encoder step={state.global_step} dir={save_dir}")
        except Exception as e:
            logger.warning(f"[SavePromptCallback] save failed: {e}")


class EncoderCheckpointCallback(TrainerCallback):
    def __init__(self, prompt_manager, encoder_save_steps: int, prompt_save_dir: str):
        self.prompt_manager = prompt_manager
        self.encoder_save_steps = int(encoder_save_steps)
        self.prompt_save_dir = prompt_save_dir
        self._saved_steps: set = set()
        self._is_rank0_cached: Optional[bool] = None

    def _check_rank0(self) -> bool:
        if self._is_rank0_cached is not None:
            return self._is_rank0_cached
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                self._is_rank0_cached = (dist.get_rank() == 0)
            else:
                self._is_rank0_cached = True
        except Exception:
            self._is_rank0_cached = True
        return self._is_rank0_cached

    def on_step_end(self, args, state, control, **kwargs):
        if self.encoder_save_steps <= 0:
            return

        step = state.global_step
        if step <= 0 or step % self.encoder_save_steps != 0:
            return

        if step in self._saved_steps:
            return

        if not self._check_rank0():
            return

        check_dir = os.path.join(self.prompt_save_dir, f"encoder_check_{step}")

        try:
            saved = self.prompt_manager.save_active_prompts(dir=check_dir)
            self._saved_steps.add(step)
            logger.info(
                "[EncoderCheckpoint] step=%d: saved %d encoder(s) to %s",
                step, len(saved), check_dir,
            )
        except Exception as e:
            logger.warning(
                "[EncoderCheckpoint] step=%d: failed to save encoder checkpoint: %s",
                step, e,
            )