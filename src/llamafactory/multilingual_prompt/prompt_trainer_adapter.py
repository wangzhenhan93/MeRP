# src/llamafactory/multilingual_prompt/prompt_trainer_adapter.py

from typing import List
import torch
import torch.nn as nn


class PromptTrainerAdapter:

    def __init__(self, prompt_manager):
        self.prompt_manager = prompt_manager

    def list_all_prompt_params(self) -> List[torch.nn.Parameter]:
        params: List[torch.nn.Parameter] = []
        seen_ids: set = set()

        model = self.prompt_manager.model
        shared_name = self.prompt_manager.SHARED_NAME

        enc = getattr(model, shared_name, None)
        if enc is not None and isinstance(enc, nn.Module):
            for p in enc.parameters():
                if id(p) not in seen_ids:
                    params.append(p)
                    seen_ids.add(id(p))

        return params

    def activate_all(self, model: nn.Module, freeze_model: bool = True) -> List[torch.nn.Parameter]:
        if freeze_model:
            for p in model.parameters():
                p.requires_grad = False

        params = self.list_all_prompt_params()
        for p in params:
            p.requires_grad = True

        return params
