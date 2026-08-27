# src/llamafactory/multilingual_prompt/prompt_encoder.py

import inspect
import logging
import math
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _zero_tensor_like_device(device: torch.device) -> torch.Tensor:
    return torch.tensor(0.0, device=device)


def _default_nhead_for_dim(dim: int) -> int:
    for n in (16, 12, 8, 4, 2, 1):
        if n <= dim and dim % n == 0:
            return n
    return 1


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: Optional[int] = None,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
        ffn_mult: float = 2.0,
    ):
        super().__init__()
        if nhead is None:
            nhead = _default_nhead_for_dim(d_model)
        if dim_feedforward is None:
            dim_feedforward = int(ffn_mult * d_model)

        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm_out = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout_ffn = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        qn = self.norm_q(q)
        kvn = self.norm_kv(kv)
        attn_out, _ = self.cross_attn(
            qn, kvn, kvn,
            key_padding_mask=src_key_padding_mask,
        )
        x = q + self.dropout1(attn_out)
        x2 = self.linear2(self.dropout_ffn(F.gelu(self.linear1(self.norm_out(x)))))
        x = x + self.dropout2(x2)
        return x


class SDANMEncoder(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        prompt_length: int = 32,
        trunk_dim: int = 256,
        num_adapters: int = 4,
        adapter_rank: int = 16,
        dir_dim: int = 64,
        num_lang_pairs: int = 10,
        dropout: float = 0.1,
        ffn_mult: float = 2.0,
        output_scale_init: float = 1.0,
        up_proj_init_std: float = 1e-4,
        enable_adapters: bool = True,
        enable_contrastive_loss: bool = True,
        enable_balance_loss: bool = True,
        enable_anchor_loss: bool = False,
        anchor_expert_by_lang_id: Optional[list[int]] = None,
        anchor_overflow_expert_ids: Optional[list[int]] = None,
        anchor_target_main_prob: float = 0.78,
        anchor_target_overflow_prob: float = 0.08,
        anchor_margin: float = 0.15,
        anchor_margin_weight: float = 0.25,
    ):
        super().__init__()

        self.embed_dim = int(embed_dim)
        self.prompt_length = int(prompt_length)
        self.trunk_dim = int(trunk_dim)
        self.num_adapters = int(num_adapters)
        self.adapter_rank = int(adapter_rank)
        self.dir_dim = int(dir_dim)
        self.num_lang_pairs = int(num_lang_pairs)
        self.dropout_rate = float(dropout)
        self.ffn_mult = float(ffn_mult)
        self.output_scale_init = float(output_scale_init)
        self.up_proj_init_std = float(up_proj_init_std)
        self.enable_adapters = bool(enable_adapters)
        self.enable_contrastive_loss = bool(enable_contrastive_loss)
        self.enable_balance_loss = bool(enable_balance_loss)
        self.enable_anchor_loss = bool(enable_anchor_loss)

        self.anchor_expert_by_lang_id = (
            [int(x) for x in anchor_expert_by_lang_id]
            if anchor_expert_by_lang_id is not None else None
        )
        self.anchor_overflow_expert_ids = sorted({
            int(x) for x in (anchor_overflow_expert_ids or [])
        })
        self.anchor_target_main_prob = float(anchor_target_main_prob)
        self.anchor_target_overflow_prob = float(anchor_target_overflow_prob)
        self.anchor_margin = float(anchor_margin)
        self.anchor_margin_weight = float(anchor_margin_weight)

        if self.num_lang_pairs < 1:
            raise ValueError(f"num_lang_pairs must be >= 1, got {self.num_lang_pairs}")
        if self.enable_anchor_loss:
            if not self.enable_adapters:
                raise ValueError("Anchor loss requires enable_adapters=True")
            if self.anchor_expert_by_lang_id is None:
                raise ValueError("enable_anchor_loss=True requires anchor_expert_by_lang_id")
            if len(self.anchor_expert_by_lang_id) != self.num_lang_pairs:
                raise ValueError(
                    "anchor_expert_by_lang_id length must equal num_lang_pairs: "
                    f"{len(self.anchor_expert_by_lang_id)} vs {self.num_lang_pairs}"
                )
            if any(idx < 0 or idx >= self.num_adapters for idx in self.anchor_expert_by_lang_id):
                raise ValueError(
                    f"anchor_expert_by_lang_id contains invalid expert ids for num_adapters={self.num_adapters}"
                )
            if any(idx < 0 or idx >= self.num_adapters for idx in self.anchor_overflow_expert_ids):
                raise ValueError(
                    f"anchor_overflow_expert_ids contains invalid expert ids for num_adapters={self.num_adapters}"
                )
            overlap = set(self.anchor_expert_by_lang_id) & set(self.anchor_overflow_expert_ids)
            if overlap:
                raise ValueError(
                    f"anchor_expert_by_lang_id overlaps with anchor_overflow_expert_ids: {sorted(overlap)}"
                )
            target_mass = self.anchor_target_main_prob + len(self.anchor_overflow_expert_ids) * self.anchor_target_overflow_prob
            if target_mass > 1.0 + 1e-8:
                raise ValueError(
                    "anchor target probabilities exceed 1.0: "
                    f"main={self.anchor_target_main_prob}, overflow={self.anchor_target_overflow_prob}, "
                    f"overflow_count={len(self.anchor_overflow_expert_ids)}"
                )

        self.in_norm = nn.LayerNorm(self.embed_dim)
        self.down_proj = nn.Linear(self.embed_dim, self.trunk_dim)

        self.shared_queries = nn.Parameter(
            torch.randn(1, self.prompt_length, self.trunk_dim) * 0.02
        )
        self.shared_cross_attn = CrossAttentionBlock(
            d_model=self.trunk_dim,
            dropout=self.dropout_rate,
            ffn_mult=self.ffn_mult,
        )

        self.direction_embed = nn.Embedding(self.num_lang_pairs, self.dir_dim)
        nn.init.normal_(self.direction_embed.weight, std=0.02)

        self.gate_proj = nn.Linear(self.dir_dim, self.num_adapters)
        nn.init.normal_(self.gate_proj.weight, std=0.02)
        nn.init.zeros_(self.gate_proj.bias)

        if self.enable_adapters:
            self.adapter_down_weight = nn.Parameter(
                torch.empty(self.num_adapters, self.adapter_rank, self.trunk_dim)
            )
            self.adapter_up_weight = nn.Parameter(
                torch.empty(self.num_adapters, self.trunk_dim, self.adapter_rank)
            )
            self._reset_adapter_weights()

        self.up_proj = nn.Linear(self.trunk_dim, self.embed_dim)
        nn.init.normal_(self.up_proj.weight, std=self.up_proj_init_std)
        nn.init.zeros_(self.up_proj.bias)

        self.out_norm = nn.LayerNorm(self.embed_dim)
        self.output_scale = nn.Parameter(
            torch.tensor(self.output_scale_init, dtype=torch.float32)
        )

        self.last_routing_weights: Optional[torch.Tensor] = None

        logger.info(
            "[SDARAEncoder] Initialized with config: "
            "embed_dim=%d, trunk_dim=%d, prompt_length=%d, "
            "num_adapters=%d, adapter_rank=%d, dir_dim=%d, "
            "num_lang_pairs=%d, expert_impl=%s",
            self.embed_dim, self.trunk_dim, self.prompt_length,
            self.num_adapters, self.adapter_rank, self.dir_dim,
            self.num_lang_pairs, "vectorized",
        )
        logger.info(
            "[SDARAEncoder] Ablation switches — "
            "Direction Adapters: %s | Contrastive Loss: %s | "
            "Balance Loss: %s | Anchor Loss: %s",
            "Enabled" if self.enable_adapters else "Disabled (Bypassed)",
            "Enabled" if self.enable_contrastive_loss else "Disabled (Bypassed)",
            "Enabled" if self.enable_balance_loss else "Disabled (Bypassed)",
            "Enabled" if self.enable_anchor_loss else "Disabled (Bypassed)",
        )

    def _reset_adapter_weights(self) -> None:
        if not self.enable_adapters:
            return

        nn.init.kaiming_uniform_(self.adapter_down_weight.view(-1, self.trunk_dim), a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.adapter_up_weight.view(-1, self.adapter_rank), a=math.sqrt(5))

    def _compute_all_adapter_outputs(self, h_shared: torch.Tensor) -> torch.Tensor:
        h_down = torch.einsum("bpt,krt->bkpr", h_shared, self.adapter_down_weight)
        h_down = F.gelu(h_down)
        return torch.einsum("bkpr,ktr->bkpt", h_down, self.adapter_up_weight)

    def forward(
        self,
        input_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        lang_pair_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if input_embeds is None or input_embeds.dim() != 3:
            raise ValueError("input_embeds must be (B, S, embed_dim)")

        ref_param = self.shared_queries
        if input_embeds.dtype != ref_param.dtype:
            input_embeds = input_embeds.to(dtype=ref_param.dtype)
        if input_embeds.device != ref_param.device:
            input_embeds = input_embeds.to(device=ref_param.device)

        bsz, seq_len, dim = input_embeds.size()
        device = input_embeds.device

        if dim != self.embed_dim:
            raise ValueError(f"Expected embed_dim={self.embed_dim}, got {dim}")

        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)

        if lang_pair_id is None:
            lang_pair_id = torch.zeros(bsz, dtype=torch.long, device=device)
        if lang_pair_id.dim() == 0:
            lang_pair_id = lang_pair_id.unsqueeze(0).expand(bsz)
        lang_pair_id = lang_pair_id.to(device)

        _invalid_mask = (lang_pair_id < 0) | (lang_pair_id >= self.num_lang_pairs)
        if _invalid_mask.any():
            _bad_ids = lang_pair_id[_invalid_mask].tolist()
            raise ValueError(
                f"[SDARAEncoder] lang_pair_id 包含越界值 {_bad_ids}。"
                f" 合法范围: [0, {self.num_lang_pairs - 1}]（共 {self.num_lang_pairs} 个方向）。"
                f" 请检查 lang_pair_map.json 与 YAML 中的"
                f" prompt_encoder_num_lang_pairs={self.num_lang_pairs} 是否一致。"
            )

        x = self.in_norm(input_embeds)
        x_low = self.down_proj(x)  # (B, S, trunk_dim)

        q = self.shared_queries.expand(bsz, -1, -1)  # (B, P, trunk_dim)
        h_shared = self.shared_cross_attn(
            q, x_low, src_key_padding_mask=src_key_padding_mask
        )  # (B, P, trunk_dim)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[SDA-RA][A-SharedTrunk] input_embeds: %s -> "
                "down_proj(x_low): %s -> h_shared: %s  "
                "(维度变换: %d -> %d -> (%d, %d, %d))",
                tuple(input_embeds.shape), tuple(x_low.shape), tuple(h_shared.shape),
                self.embed_dim, self.trunk_dim,
                bsz, self.prompt_length, self.trunk_dim,
            )

        e_dir = self.direction_embed(lang_pair_id)  # (B, dir_dim)

        if self.enable_adapters:
            g = torch.softmax(self.gate_proj(e_dir), dim=-1)  # (B, K)
            self.last_routing_weights = g.detach()

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "[SDA-RA][B-Router] Activated — "
                    "lang_pair_ids: %s | g shape: %s | "
                    "g mean: %.4f | g top-1 expert per sample: %s",
                    lang_pair_id.tolist(),
                    tuple(g.shape),
                    g.mean().item(),
                    g.argmax(dim=-1).tolist(),
                )

            adapter_out_all = self._compute_all_adapter_outputs(h_shared)  # (B, K, P, trunk_dim)
            h_private = torch.einsum("bk,bkpt->bpt", g, adapter_out_all)

            h_out = h_shared + h_private
        else:
            h_out = h_shared
            self.last_routing_weights = None
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "[SDA-RA][B-Router] Bypassed (enable_adapters=False) — "
                    "h_out = h_shared directly. shape: %s",
                    tuple(h_out.shape),
                )

        out = self.up_proj(h_out)  # (B, P, embed_dim)
        out = self.out_norm(out)
        s = self.output_scale.float().clamp(0.01, 10.0).to(dtype=out.dtype, device=out.device)
        out = out * s

        return out

    def compute_contrastive_loss(self) -> torch.Tensor:
        if not self.enable_contrastive_loss or not self.enable_adapters:
            return _zero_tensor_like_device(self.direction_embed.weight.device)

        all_ids = torch.arange(self.num_lang_pairs, device=self.direction_embed.weight.device)
        e_all = self.direction_embed(all_ids)  # (N, dir_dim)
        g_all = torch.softmax(self.gate_proj(e_all), dim=-1)  # (N, K)

        g_norm = F.normalize(g_all, p=2, dim=-1)  # (N, K)
        cos_sim = g_norm @ g_norm.t()  # (N, N)

        mask = 1.0 - torch.eye(self.num_lang_pairs, device=cos_sim.device)
        num_pairs = self.num_lang_pairs * (self.num_lang_pairs - 1)
        loss = (cos_sim * mask).sum() / max(num_pairs, 1)
        return loss

    def _compute_all_router_logits_and_probs(self) -> tuple[torch.Tensor, torch.Tensor]:
        all_ids = torch.arange(self.num_lang_pairs, device=self.direction_embed.weight.device)
        e_all = self.direction_embed(all_ids)
        logits = self.gate_proj(e_all).float()
        probs = torch.softmax(logits, dim=-1)
        return logits, probs

    def compute_balance_loss(self) -> torch.Tensor:
        if not self.enable_balance_loss or not self.enable_adapters:
            return _zero_tensor_like_device(self.direction_embed.weight.device)

        _, g_all = self._compute_all_router_logits_and_probs()

        f_k = g_all.mean(dim=0)  # (K,)
        K = float(self.num_adapters)
        loss = K * (f_k * f_k).sum()
        return loss

    def compute_anchor_loss(self) -> torch.Tensor:
        if not self.enable_anchor_loss or not self.enable_adapters:
            return _zero_tensor_like_device(self.direction_embed.weight.device)

        logits, g_all = self._compute_all_router_logits_and_probs()  # (N, K)
        device = logits.device
        dtype = logits.dtype

        anchor_ids = torch.tensor(self.anchor_expert_by_lang_id, device=device, dtype=torch.long)
        overflow_ids = torch.tensor(self.anchor_overflow_expert_ids, device=device, dtype=torch.long)

        target = torch.zeros_like(g_all, dtype=dtype)
        target.scatter_(1, anchor_ids.unsqueeze(1), float(self.anchor_target_main_prob))

        if overflow_ids.numel() > 0:
            target[:, overflow_ids] = float(self.anchor_target_overflow_prob)

        forbidden = torch.zeros_like(target, dtype=torch.bool)
        forbidden.scatter_(1, anchor_ids.unsqueeze(1), True)
        if overflow_ids.numel() > 0:
            forbidden[:, overflow_ids] = True

        remaining_mass = 1.0 - float(self.anchor_target_main_prob) - float(self.anchor_target_overflow_prob) * int(overflow_ids.numel())
        remaining_count = (~forbidden).sum(dim=1, keepdim=True).clamp_min(1)
        if remaining_mass < -1e-8:
            raise ValueError(f"anchor target mass becomes negative: {remaining_mass}")
        if remaining_mass > 0:
            target = target + (~forbidden).to(dtype) * (remaining_mass / remaining_count.to(dtype))

        log_probs = torch.log(g_all.clamp_min(1e-8))
        anchor_ce = -(target * log_probs).sum(dim=-1).mean()

        competitor_logits = logits.masked_fill(
            torch.nn.functional.one_hot(anchor_ids, num_classes=self.num_adapters).bool(),
            float("-inf"),
        )
        best_other = competitor_logits.max(dim=-1).values
        anchor_logits = logits.gather(1, anchor_ids.unsqueeze(1)).squeeze(1)
        margin_loss = F.relu(float(self.anchor_margin) - (anchor_logits - best_other)).mean()

        return anchor_ce + float(self.anchor_margin_weight) * margin_loss

    def get_config(self) -> Dict[str, Any]:
        return {
            "encoder_type": "sda_ra",
            "embed_dim": self.embed_dim,
            "prompt_length": self.prompt_length,
            "trunk_dim": self.trunk_dim,
            "num_adapters": self.num_adapters,
            "adapter_rank": self.adapter_rank,
            "dir_dim": self.dir_dim,
            "num_lang_pairs": self.num_lang_pairs,
            "dropout": self.dropout_rate,
            "ffn_mult": self.ffn_mult,
            "output_scale_init": self.output_scale_init,
            "up_proj_init_std": self.up_proj_init_std,
            "enable_adapters": self.enable_adapters,
            "enable_contrastive_loss": self.enable_contrastive_loss,
            "enable_balance_loss": self.enable_balance_loss,
            "enable_anchor_loss": self.enable_anchor_loss,
            "anchor_expert_by_lang_id": list(self.anchor_expert_by_lang_id) if self.anchor_expert_by_lang_id is not None else None,
            "anchor_overflow_expert_ids": list(self.anchor_overflow_expert_ids),
            "anchor_target_main_prob": self.anchor_target_main_prob,
            "anchor_target_overflow_prob": self.anchor_target_overflow_prob,
            "anchor_margin": self.anchor_margin,
            "anchor_margin_weight": self.anchor_margin_weight,
        }

    def get_routing_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {}
        with torch.no_grad():
            all_ids = torch.arange(
                self.num_lang_pairs,
                device=self.direction_embed.weight.device,
            )
            e_all = self.direction_embed(all_ids)

            if self.enable_adapters:
                g_all = torch.softmax(self.gate_proj(e_all), dim=-1)
                info["routing_matrix"] = g_all.cpu()  # (N, K)
                info["top1_expert_per_dir"] = g_all.argmax(dim=-1).cpu()
                info["top1_prob_per_dir"] = g_all.max(dim=-1).values.cpu()
                info["routing_entropy_per_dir"] = (-(g_all * torch.log(g_all.clamp_min(1e-8))).sum(dim=-1)).cpu()
                if self.enable_anchor_loss and self.anchor_expert_by_lang_id is not None:
                    logits = self.gate_proj(e_all).float()
                    anchor_ids = torch.tensor(self.anchor_expert_by_lang_id, device=g_all.device, dtype=torch.long)
                    info["anchor_expert_by_lang_id"] = anchor_ids.cpu()
                    info["anchor_prob_per_dir"] = g_all.gather(1, anchor_ids.unsqueeze(1)).squeeze(1).cpu()
                    if self.anchor_overflow_expert_ids:
                        overflow_ids = torch.tensor(self.anchor_overflow_expert_ids, device=g_all.device, dtype=torch.long)
                        info["anchor_overflow_expert_ids"] = overflow_ids.cpu()
                        info["overflow_prob_per_dir"] = g_all.index_select(dim=1, index=overflow_ids).sum(dim=1).cpu()
                    competitor_logits = logits.masked_fill(
                        torch.nn.functional.one_hot(anchor_ids, num_classes=self.num_adapters).bool(),
                        float("-inf"),
                    )
                    best_other = competitor_logits.max(dim=-1).values
                    anchor_logits = logits.gather(1, anchor_ids.unsqueeze(1)).squeeze(1)
                    info["anchor_margin_per_dir"] = (anchor_logits - best_other).cpu()

        if self.last_routing_weights is not None:
            info["last_batch_routing"] = self.last_routing_weights.cpu()

        return info


def build_from_config(config: Optional[Dict[str, Any]], embed_dim: int) -> nn.Module:
    cfg = dict(config or {})
    cfg.pop("embed_dim", None)
    encoder_type = str(cfg.pop("encoder_type", "sda_ra")).lower()

    if encoder_type == "sda_ra":
        valid_keys = set(inspect.signature(SDANMEncoder.__init__).parameters.keys())
        valid_keys.discard("self")
        valid_keys.discard("embed_dim")

        filtered_cfg = {k: v for k, v in cfg.items() if k in valid_keys}
        dropped_keys = sorted(k for k in cfg.keys() if k not in valid_keys)
        if dropped_keys:
            logger.warning(
                "[build_from_config] Ignore non-constructor checkpoint config keys: %s",
                dropped_keys,
            )

        return SDANMEncoder(embed_dim=embed_dim, **filtered_cfg)

    raise ValueError(
        f"Unsupported encoder_type='{encoder_type}'. Only 'sda_ra' is supported."
    )


def build_from_finetuning_args(finetuning_args, embed_dim: int) -> nn.Module:
    encoder_type = str(
        getattr(finetuning_args, "prompt_encoder_type", "sda_ra")
    ).lower()

    if encoder_type == "sda_ra":
        return SDANMEncoder(
            embed_dim=embed_dim,
            prompt_length=int(getattr(finetuning_args, "prompt_encoder_prompt_length", 32)),
            trunk_dim=int(getattr(finetuning_args, "sda_trunk_dim", 256)),
            num_adapters=int(getattr(finetuning_args, "sda_num_adapters", 4)),
            adapter_rank=int(getattr(finetuning_args, "sda_adapter_rank", 16)),
            dir_dim=int(getattr(finetuning_args, "sda_dir_dim", 64)),
            num_lang_pairs=int(getattr(finetuning_args, "prompt_encoder_num_lang_pairs", 10)),
            dropout=float(getattr(finetuning_args, "prompt_encoder_dropout", 0.1)),
            ffn_mult=float(getattr(finetuning_args, "prompt_encoder_ffn_mult", 2.0)),
            output_scale_init=float(getattr(finetuning_args, "prompt_encoder_output_scale", 1.0)),
            up_proj_init_std=float(getattr(finetuning_args, "sda_up_proj_init_std", 5e-4)),
            enable_adapters=bool(getattr(finetuning_args, "sda_enable_adapters", True)),
            enable_contrastive_loss=bool(getattr(finetuning_args, "sda_enable_contrastive_loss", True)),
            enable_balance_loss=bool(getattr(finetuning_args, "sda_enable_balance_loss", True)),
            enable_anchor_loss=bool(getattr(finetuning_args, "sda_enable_anchor_loss", False)),
            anchor_expert_by_lang_id=getattr(finetuning_args, "_parsed_sda_anchor_expert_by_lang_id", None),
            anchor_overflow_expert_ids=getattr(finetuning_args, "_parsed_sda_anchor_overflow_expert_ids", None),
            anchor_target_main_prob=float(getattr(finetuning_args, "sda_anchor_target_main_prob", 0.78)),
            anchor_target_overflow_prob=float(getattr(finetuning_args, "sda_anchor_target_overflow_prob", 0.08)),
            anchor_margin=float(getattr(finetuning_args, "sda_anchor_margin", 0.15)),
            anchor_margin_weight=float(getattr(finetuning_args, "sda_anchor_margin_weight", 0.25)),
        )

    raise ValueError(
        f"Unsupported prompt_encoder_type='{encoder_type}'. Only 'sda_ra' is supported."
    )
