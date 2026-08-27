# src/llamafactory/multilingual_prompt/prompt_injector.py

import logging
from typing import Tuple, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def attach_prompt_to_inputs_embeds(
    inputs_embeds: torch.Tensor,
    prompt_tensor: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:

    if device is None:
        device = inputs_embeds.device

    if inputs_embeds.dim() != 3:
        raise ValueError(f"inputs_embeds must be (B, T, D), got {inputs_embeds.shape}")
    if prompt_tensor.dim() != 3:
        raise ValueError(f"prompt_tensor must be (B, P, D), got {prompt_tensor.shape}")

    B, T, D = inputs_embeds.shape
    if prompt_tensor.size(0) != B:
        raise ValueError(
            f"Batch size mismatch: inputs_embeds={B}, prompt_tensor={prompt_tensor.size(0)}"
        )
    if prompt_tensor.size(2) != D:
        raise ValueError(
            f"Embed dim mismatch: inputs_embeds D={D}, prompt_tensor D={prompt_tensor.size(2)}"
        )

    P = prompt_tensor.size(1)
    new_inputs = torch.cat([prompt_tensor.to(device), inputs_embeds.to(device)], dim=1)

    if attention_mask is None:
        new_mask = torch.ones((B, P + T), dtype=torch.long, device=device)
    else:
        prefix = torch.ones((B, P), dtype=attention_mask.dtype, device=device)
        new_mask = torch.cat([prefix, attention_mask.to(device)], dim=1)

    return new_inputs, new_mask


def inject_prompt(
    model: nn.Module,
    prompt_manager,
    lang_pair: str,
    input_ids: Optional[torch.Tensor] = None,
    input_embeds: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    encoder_input_ids: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    lang_pair_id: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if device is None:
        if input_embeds is not None:
            device = input_embeds.device
        elif input_ids is not None:
            device = input_ids.device
        else:
            device = next(model.parameters()).device

    if input_embeds is None:
        if input_ids is None:
            raise ValueError("Either input_embeds or input_ids must be provided")
        input_embeds = model.get_input_embeddings()(input_ids.to(device))

    if input_embeds.dim() != 3:
        raise ValueError(f"input_embeds must be (B, T, D), got {input_embeds.shape}")

    encoder = prompt_manager.get_prompt(lang_pair)
    if not isinstance(encoder, nn.Module):
        raise TypeError(f"Expected nn.Module encoder for '{lang_pair}', got {type(encoder)}")

    if encoder_input_ids is None:
        raise ValueError(
            f"encoder_input_ids required for SDA-RA. "
            f"Collator must provide encoder_input_ids from source text."
        )

    enc_embeds = model.get_input_embeddings()(encoder_input_ids.to(device))
    if encoder_attention_mask is None:
        encoder_attention_mask = torch.ones(
            encoder_input_ids.shape[:2], dtype=torch.long, device=device
        )
    else:
        encoder_attention_mask = encoder_attention_mask.to(device)

    prompt_embeds = encoder(
        enc_embeds,
        encoder_attention_mask,
        lang_pair_id=lang_pair_id,
    )

    if prompt_embeds.dim() != 3:
        raise RuntimeError(
            f"Encoder returned invalid shape {prompt_embeds.shape}, expected (B, P, D)"
        )

    new_inputs, new_mask = attach_prompt_to_inputs_embeds(
        input_embeds, prompt_embeds, attention_mask, device=device
    )

    logger.debug(
        "[inject_prompt] lang=%s prompt=%s final=%s",
        lang_pair, tuple(prompt_embeds.shape), tuple(new_inputs.shape),
    )

    return new_inputs, new_mask
