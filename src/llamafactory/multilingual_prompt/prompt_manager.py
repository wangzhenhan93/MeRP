# src/llamafactory/multilingual_prompt/prompt_manager.py

import os
import json
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone

import torch
import torch.nn as nn

from ..data.parser import normalize_lang_pair
from .prompt_encoder import build_from_finetuning_args, build_from_config

logger = logging.getLogger(__name__)


class PromptManager:
    SHARED_NAME = "prompt_enc_shared"

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        save_dir: str = "prompts",
        args: Optional[Any] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.args = args

        try:
            self.D = model.get_input_embeddings().weight.shape[1]
        except Exception:
            raise RuntimeError("Model must expose get_input_embeddings()")

        self.encoders: Dict[str, str] = {}
        self.metadata: Dict[str, dict] = {}
        self.active_lang_pairs: set = set()
        self._encoder_created: bool = False
        self.configured_lang_pairs: List[str] = [
            normalize_lang_pair(lp) for lp in (getattr(args, "lang_pairs", None) or [])
        ]
        self.lang_pair_to_id: Dict[str, int] = {
            lp: i for i, lp in enumerate(self.configured_lang_pairs)
        }

        logger.info(
            "PromptManager initialized. save_dir=%s, embed_dim=%d, shared_encoder=%s",
            self.save_dir, int(self.D), self.SHARED_NAME,
        )

    def _final_dir(self, base_dir: Optional[str] = None) -> str:
        return os.path.join(base_dir or self.save_dir, "finall")

    def _checkpoint_path(self, base_dir: Optional[str] = None) -> str:
        final_dir = self._final_dir(base_dir)
        os.makedirs(final_dir, exist_ok=True)
        return os.path.join(final_dir, f"{self.SHARED_NAME}.pt")

    def _checkpoint_candidates(self) -> List[str]:
        final_dir = self._final_dir()
        return [
            os.path.join(final_dir, f"{self.SHARED_NAME}.pt"),
            os.path.join(final_dir, f"{self.SHARED_NAME}.pth"),
            os.path.join(self.save_dir, f"{self.SHARED_NAME}.pt"),
            os.path.join(self.save_dir, f"{self.SHARED_NAME}.pth"),
        ]

    def _ensure_shared_encoder(self, save: bool = False) -> nn.Module:
        if self._encoder_created:
            return getattr(self.model, self.SHARED_NAME)

        if self.args is None:
            raise RuntimeError("Cannot create encoder without training args.")

        encoder = build_from_finetuning_args(self.args, embed_dim=self.D)

        if self.SHARED_NAME in getattr(self.model, "_modules", {}):
            del self.model._modules[self.SHARED_NAME]
        setattr(self.model, self.SHARED_NAME, encoder)
        self._encoder_created = True

        logger.info(
            "Created shared SDA-RA encoder -> %s, config=%s",
            self.SHARED_NAME, encoder.get_config(),
        )

        if save:
            self._save_shared_encoder()

        return encoder

    def _load_shared_encoder(self, path: str) -> nn.Module:
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        if self._encoder_created:
            return getattr(self.model, self.SHARED_NAME)

        ckpt = torch.load(path, map_location="cpu")
        if "encoder_state_dict" not in ckpt:
            raise RuntimeError(f"{path} is not a valid encoder checkpoint")

        cfg = ckpt.get("config", {}) or {}
        meta = ckpt.get("metadata", {}) or {}
        encoder = build_from_config(cfg, embed_dim=self.D) if cfg else build_from_finetuning_args(self.args, embed_dim=self.D)
        state_dict = dict(ckpt["encoder_state_dict"])

        try:
            encoder.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                "Failed to load shared SDA-RA encoder strictly. "
                "Please ensure training/inference config exactly matches the checkpoint structure. "
                f"checkpoint={path}"
            ) from exc

        ckpt_pairs = meta.get("lang_pairs") or cfg.get("lang_pairs") or []
        if ckpt_pairs:
            self.configured_lang_pairs = [normalize_lang_pair(lp) for lp in ckpt_pairs]
        ckpt_map = meta.get("lang_pair_to_id") or {}
        if ckpt_map:
            self.lang_pair_to_id = {normalize_lang_pair(str(k)): int(v) for k, v in ckpt_map.items()}
        elif self.configured_lang_pairs:
            self.lang_pair_to_id = {lp: i for i, lp in enumerate(self.configured_lang_pairs)}
        self.metadata["_shared"] = meta

        if self.SHARED_NAME in getattr(self.model, "_modules", {}):
            del self.model._modules[self.SHARED_NAME]
        setattr(self.model, self.SHARED_NAME, encoder)
        self._encoder_created = True

        logger.info("Loaded shared SDA-RA encoder from %s", path)
        return encoder

    def _save_shared_encoder(self, base_dir: Optional[str] = None) -> str:
        encoder = getattr(self.model, self.SHARED_NAME)
        path = self._checkpoint_path(base_dir)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        ordered_lang_pairs = list(self.configured_lang_pairs)
        if not ordered_lang_pairs:
            ordered_lang_pairs = sorted(self.active_lang_pairs)
        lang_pair_to_id = {lp: i for i, lp in enumerate(ordered_lang_pairs)}

        encoder_state_dict = dict(encoder.state_dict())
        suspicious_keys = [
            key for key in encoder_state_dict.keys()
            if key.startswith(("model.", "base_model.", "pretrained_model.", "language_model."))
        ]
        if suspicious_keys:
            raise RuntimeError(
                "PromptManager detected non-SDA-RA keys while saving shared encoder: "
                f"{suspicious_keys[:10]}"
            )

        payload = {
            "encoder_state_dict": encoder_state_dict,
            "config": {
                **(encoder.get_config() if hasattr(encoder, "get_config") else {}),
                "lang_pairs": ordered_lang_pairs,
            },
            "metadata": {
                "type": "sda_ra_shared",
                "lang_pairs": ordered_lang_pairs,
                "active_lang_pairs": sorted(self.active_lang_pairs),
                "lang_pair_to_id": lang_pair_to_id,
                "saved_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            },
        }
        torch.save(payload, path)

        try:
            with open(path + ".meta.json", "w", encoding="utf-8") as f:
                json.dump(payload["metadata"], f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        logger.info("Saved shared SDA-RA encoder -> %s", path)
        return path

    def resolve_lang_pair_id(self, lang_pair: str) -> Optional[int]:
        norm_lp = normalize_lang_pair(lang_pair)
        if norm_lp in self.lang_pair_to_id:
            return int(self.lang_pair_to_id[norm_lp])
        return None

    def create_encoder(self, lang_pair: str, save: bool = False) -> nn.Module:
        norm_lp = normalize_lang_pair(lang_pair)
        encoder = self._ensure_shared_encoder(save=save)
        self.encoders[norm_lp] = self.SHARED_NAME
        self.metadata[norm_lp] = {
            "lang_pair": norm_lp,
            "type": "sda_ra_shared",
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        return encoder

    def load_prompt(self, lang_pair: str, path: str) -> nn.Module:
        norm_lp = normalize_lang_pair(lang_pair)
        encoder = self._load_shared_encoder(path)
        self.encoders[norm_lp] = self.SHARED_NAME
        self.metadata[norm_lp] = {"lang_pair": norm_lp, "loaded_from": path}
        return encoder

    def save_prompt(self, lang_pair: str, path: Optional[str] = None, metadata: Optional[dict] = None) -> str:
        norm_lp = normalize_lang_pair(lang_pair)
        if norm_lp not in self.encoders:
            raise KeyError(f"No encoder registered for {norm_lp}")
        return self._save_shared_encoder()

    def get_prompt(self, lang_pair: str) -> nn.Module:
        norm_lp = normalize_lang_pair(lang_pair)
        if norm_lp not in self.encoders:
            raise KeyError(f"Encoder for {norm_lp} not registered. Call ensure_prompts first.")
        self.active_lang_pairs.add(norm_lp)
        return getattr(self.model, self.SHARED_NAME)

    def list_prompts(self) -> List[str]:
        return list(self.encoders.keys())

    def ensure_prompts(self, lang_list: List[str]) -> Dict[str, nn.Module]:
        result = {}
        is_training = bool(getattr(self.args, "save_prompts", False))

        for lp in lang_list:
            norm_lp = normalize_lang_pair(lp)

            if norm_lp in self.encoders:
                result[norm_lp] = self.get_prompt(norm_lp)
                continue

            if not self._encoder_created:
                loaded = False
                for cand in self._checkpoint_candidates():
                    if os.path.exists(cand):
                        result[norm_lp] = self.load_prompt(norm_lp, cand)
                        loaded = True
                        break
                if loaded:
                    continue

            if self._encoder_created:
                self.encoders[norm_lp] = self.SHARED_NAME
                self.active_lang_pairs.add(norm_lp)
                result[norm_lp] = getattr(self.model, self.SHARED_NAME)
                continue

            if not is_training:
                candidates = self._checkpoint_candidates()
                raise FileNotFoundError(
                    f"Encoder checkpoint not found during inference.\n"
                    f"Searched: {candidates}\n"
                    f"Check prompt_save_dir='{self.save_dir}'."
                )
            result[norm_lp] = self.create_encoder(norm_lp, save=False)

        return result

    def save_active_prompts(self, dir: Optional[str] = None) -> Dict[str, str]:
        if not self._encoder_created:
            return {}
        path = self._save_shared_encoder(dir)
        return {lp: path for lp in self.active_lang_pairs}

    def save_prompts(self, save_dir: Optional[str] = None, langs=None) -> Dict[str, str]:
        if not self._encoder_created:
            return {}
        path = self._save_shared_encoder(save_dir)
        return {"shared": path}
