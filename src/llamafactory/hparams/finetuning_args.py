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
# src/llamafactory/hparams/finetuning_args.py
from dataclasses import dataclass, field
from typing import List, Optional
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional


@dataclass
class FreezeArguments:
    r"""Arguments pertaining to the freeze (partial-parameter) training."""

    freeze_trainable_layers: int = field(
        default=2,
        metadata={
            "help": (
                "The number of trainable layers for freeze (partial-parameter) fine-tuning. "
                "Positive numbers mean the last n layers are set as trainable, "
                "negative numbers mean the first n layers are set as trainable."
            )
        },
    )
    freeze_trainable_modules: str = field(
        default="all",
        metadata={
            "help": (
                "Name(s) of trainable modules for freeze (partial-parameter) fine-tuning. "
                "Use commas to separate multiple modules. "
                "Use `all` to specify all the available modules."
            )
        },
    )
    freeze_extra_modules: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Name(s) of modules apart from hidden layers to be set as trainable "
                "for freeze (partial-parameter) fine-tuning. "
                "Use commas to separate multiple modules."
            )
        },
    )


@dataclass
class LoraArguments:
    r"""Arguments pertaining to the LoRA training."""

    additional_target: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Name(s) of modules apart from LoRA layers to be set as trainable "
                "and saved in the final checkpoint. "
                "Use commas to separate multiple modules."
            )
        },
    )
    lora_alpha: Optional[int] = field(
        default=None,
        metadata={"help": "The scale factor for LoRA fine-tuning (default: lora_rank * 2)."},
    )
    lora_dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout rate for the LoRA fine-tuning."},
    )
    lora_rank: int = field(
        default=8,
        metadata={"help": "The intrinsic dimension for LoRA fine-tuning."},
    )
    lora_target: str = field(
        default="all",
        metadata={
            "help": (
                "Name(s) of target modules to apply LoRA. "
                "Use commas to separate multiple modules. "
                "Use `all` to specify all the linear modules."
            )
        },
    )
    loraplus_lr_ratio: Optional[float] = field(
        default=None,
        metadata={"help": "LoRA plus learning rate ratio (lr_B / lr_A)."},
    )
    loraplus_lr_embedding: float = field(
        default=1e-6,
        metadata={"help": "LoRA plus learning rate for lora embedding layers."},
    )
    use_rslora: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the rank stabilization scaling factor for LoRA layer."},
    )
    use_dora: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the weight-decomposed lora method (DoRA)."},
    )
    pissa_init: bool = field(
        default=False,
        metadata={"help": "Whether or not to initialize a PiSSA adapter."},
    )
    pissa_iter: int = field(
        default=16,
        metadata={"help": "The number of iteration steps performed by FSVD in PiSSA. Use -1 to disable it."},
    )
    pissa_convert: bool = field(
        default=False,
        metadata={"help": "Whether or not to convert the PiSSA adapter to a normal LoRA adapter."},
    )
    create_new_adapter: bool = field(
        default=False,
        metadata={"help": "Whether or not to create a new adapter with randomly initialized weight."},
    )


@dataclass
class OFTArguments:
    r"""Arguments pertaining to the OFT training."""

    additional_target: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Name(s) of modules apart from LoRA layers to be set as trainable "
                "and saved in the final checkpoint. "
                "Use commas to separate multiple modules."
            )
        },
    )
    module_dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout rate for the OFT fine-tuning."},
    )
    oft_rank: int = field(
        default=0,
        metadata={"help": "The intrinsic dimension for OFT fine-tuning."},
    )
    oft_block_size: int = field(
        default=32,
        metadata={"help": "The intrinsic dimension for OFT fine-tuning."},
    )
    oft_target: str = field(
        default="all",
        metadata={
            "help": (
                "Name(s) of target modules to apply OFT. "
                "Use commas to separate multiple modules. "
                "Use `all` to specify all the linear modules."
            )
        },
    )
    create_new_adapter: bool = field(
        default=False,
        metadata={"help": "Whether or not to create a new adapter with randomly initialized weight."},
    )


@dataclass
class RLHFArguments:
    r"""Arguments pertaining to the PPO, DPO and KTO training."""

    pref_beta: float = field(
        default=0.1,
        metadata={"help": "The beta parameter in the preference loss."},
    )
    pref_ftx: float = field(
        default=0.0,
        metadata={"help": "The supervised fine-tuning loss coefficient in DPO training."},
    )
    pref_bco_weight: float = field(
        default=0.0,
        metadata={"help": "The Binary Classifier Optimization coefficient in DPO training."},
    )
    pref_loss: Literal["sigmoid", "hinge", "ipo", "kto_pair", "orpo", "simpo"] = field(
        default="sigmoid",
        metadata={"help": "The type of DPO loss to use."},
    )
    dpo_label_smoothing: float = field(
        default=0.0,
        metadata={"help": "The robust DPO label smoothing parameter in cDPO that should be between 0 and 0.5."},
    )
    kto_chosen_weight: float = field(
        default=1.0,
        metadata={"help": "The weight factor of the desirable losses in KTO training."},
    )
    kto_rejected_weight: float = field(
        default=1.0,
        metadata={"help": "The weight factor of the undesirable losses in KTO training."},
    )
    simpo_gamma: float = field(
        default=0.5,
        metadata={"help": "The target reward margin term in SimPO loss."},
    )
    ppo_buffer_size: int = field(
        default=1,
        metadata={"help": "The number of mini-batches to make experience buffer in a PPO optimization step."},
    )
    ppo_epochs: int = field(
        default=4,
        metadata={"help": "The number of epochs to perform in a PPO optimization step."},
    )
    ppo_score_norm: bool = field(
        default=False,
        metadata={"help": "Use score normalization in PPO training."},
    )
    ppo_target: float = field(
        default=6.0,
        metadata={"help": "Target KL value for adaptive KL control in PPO training."},
    )
    ppo_whiten_rewards: bool = field(
        default=False,
        metadata={"help": "Whiten the rewards before compute advantages in PPO training."},
    )
    ref_model: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the reference model used for the PPO or DPO training."},
    )
    ref_model_adapters: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the adapters of the reference model."},
    )
    ref_model_quantization_bit: Optional[int] = field(
        default=None,
        metadata={"help": "The number of bits to quantize the reference model."},
    )
    reward_model: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the reward model used for the PPO training."},
    )
    reward_model_adapters: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the adapters of the reward model."},
    )
    reward_model_quantization_bit: Optional[int] = field(
        default=None,
        metadata={"help": "The number of bits to quantize the reward model."},
    )
    reward_model_type: Literal["lora", "full", "api"] = field(
        default="lora",
        metadata={"help": "The type of the reward model in PPO training. Lora model only supports lora training."},
    )
    ld_alpha: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Alpha parameter from the LD-DPO paper, which controls the weighting of"
                " the verbose token log-probabilities in responses."
            )
        },
    )


@dataclass
class GaloreArguments:
    r"""Arguments pertaining to the GaLore algorithm."""

    use_galore: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the gradient low-Rank projection (GaLore)."},
    )
    galore_target: str = field(
        default="all",
        metadata={
            "help": (
                "Name(s) of modules to apply GaLore. Use commas to separate multiple modules. "
                "Use `all` to specify all the linear modules."
            )
        },
    )
    galore_rank: int = field(
        default=16,
        metadata={"help": "The rank of GaLore gradients."},
    )
    galore_update_interval: int = field(
        default=200,
        metadata={"help": "Number of steps to update the GaLore projection."},
    )
    galore_scale: float = field(
        default=2.0,
        metadata={"help": "GaLore scaling coefficient."},
    )
    galore_proj_type: Literal["std", "reverse_std", "right", "left", "full"] = field(
        default="std",
        metadata={"help": "Type of GaLore projection."},
    )
    galore_layerwise: bool = field(
        default=False,
        metadata={"help": "Whether or not to enable layer-wise update to further save memory."},
    )


@dataclass
class ApolloArguments:
    r"""Arguments pertaining to the APOLLO algorithm."""

    use_apollo: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the APOLLO optimizer."},
    )
    apollo_target: str = field(
        default="all",
        metadata={
            "help": (
                "Name(s) of modules to apply APOLLO. Use commas to separate multiple modules. "
                "Use `all` to specify all the linear modules."
            )
        },
    )
    apollo_rank: int = field(
        default=16,
        metadata={"help": "The rank of APOLLO gradients."},
    )
    apollo_update_interval: int = field(
        default=200,
        metadata={"help": "Number of steps to update the APOLLO projection."},
    )
    apollo_scale: float = field(
        default=32.0,
        metadata={"help": "APOLLO scaling coefficient."},
    )
    apollo_proj: Literal["svd", "random"] = field(
        default="random",
        metadata={"help": "Type of APOLLO low-rank projection algorithm (svd or random)."},
    )
    apollo_proj_type: Literal["std", "right", "left"] = field(
        default="std",
        metadata={"help": "Type of APOLLO projection."},
    )
    apollo_scale_type: Literal["channel", "tensor"] = field(
        default="channel",
        metadata={"help": "Type of APOLLO scaling (channel or tensor)."},
    )
    apollo_layerwise: bool = field(
        default=False,
        metadata={"help": "Whether or not to enable layer-wise update to further save memory."},
    )
    apollo_scale_front: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the norm-growth limiter in front of gradient scaling."},
    )


@dataclass
class BAdamArgument:
    r"""Arguments pertaining to the BAdam optimizer."""

    use_badam: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the BAdam optimizer."},
    )
    badam_mode: Literal["layer", "ratio"] = field(
        default="layer",
        metadata={"help": "Whether to use layer-wise or ratio-wise BAdam optimizer."},
    )
    badam_start_block: Optional[int] = field(
        default=None,
        metadata={"help": "The starting block index for layer-wise BAdam."},
    )
    badam_switch_mode: Optional[Literal["ascending", "descending", "random", "fixed"]] = field(
        default="ascending",
        metadata={"help": "the strategy of picking block to update for layer-wise BAdam."},
    )
    badam_switch_interval: Optional[int] = field(
        default=50,
        metadata={
            "help": "Number of steps to update the block for layer-wise BAdam. Use -1 to disable the block update."
        },
    )
    badam_update_ratio: float = field(
        default=0.05,
        metadata={"help": "The ratio of the update for ratio-wise BAdam."},
    )
    badam_mask_mode: Literal["adjacent", "scatter"] = field(
        default="adjacent",
        metadata={
            "help": (
                "The mode of the mask for BAdam optimizer. "
                "`adjacent` means that the trainable parameters are adjacent to each other, "
                "`scatter` means that trainable parameters are randomly choosed from the weight."
            )
        },
    )
    badam_verbose: int = field(
        default=0,
        metadata={
            "help": (
                "The verbosity level of BAdam optimizer. "
                "0 for no print, 1 for print the block prefix, 2 for print trainable parameters."
            )
        },
    )
    


@dataclass
class SwanLabArguments:
    use_swanlab: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the SwanLab (an experiment tracking and visualization tool)."},
    )
    swanlab_project: Optional[str] = field(
        default="llamafactory",
        metadata={"help": "The project name in SwanLab."},
    )
    swanlab_workspace: Optional[str] = field(
        default=None,
        metadata={"help": "The workspace name in SwanLab."},
    )
    swanlab_run_name: Optional[str] = field(
        default=None,
        metadata={"help": "The experiment name in SwanLab."},
    )
    swanlab_mode: Literal["cloud", "local"] = field(
        default="cloud",
        metadata={"help": "The mode of SwanLab."},
    )
    swanlab_api_key: Optional[str] = field(
        default=None,
        metadata={"help": "The API key for SwanLab."},
    )
    swanlab_logdir: Optional[str] = field(
        default=None,
        metadata={"help": "The log directory for SwanLab."},
    )
    swanlab_lark_webhook_url: Optional[str] = field(
        default=None,
        metadata={"help": "The Lark(飞书) webhook URL for SwanLab."},
    )
    swanlab_lark_secret: Optional[str] = field(
        default=None,
        metadata={"help": "The Lark(飞书) secret for SwanLab."},
    )


@dataclass
class FinetuningArguments(
    SwanLabArguments,
    BAdamArgument,
    ApolloArguments,
    GaloreArguments,
    RLHFArguments,
    LoraArguments,
    OFTArguments,
    FreezeArguments,
):
    r"""Arguments pertaining to which techniques we are going to fine-tuning with."""

    pure_bf16: bool = field(
        default=False,
        metadata={"help": "Whether or not to train model in purely bf16 precision (without AMP)."},
    )
    stage: Literal["pt", "sft", "rm", "ppo", "dpo", "kto"] = field(
        default="sft",
        metadata={"help": "Which stage will be performed in training."},
    )
    finetuning_type: Literal["lora", "freeze", "full"] = field(
        default="lora",
        metadata={"help": "Which fine-tuning method to use."},
    )
    use_llama_pro: bool = field(
        default=False,
        metadata={"help": "Whether or not to make only the parameters in the expanded blocks trainable."},
    )
    use_adam_mini: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the Adam-mini optimizer."},
    )
    use_muon: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the Muon optimizer."},
    )
    use_dft_loss: bool = field(
        default=False,
        metadata={"help": "Whether to use the DFT loss."},
    )
    freeze_vision_tower: bool = field(
        default=True,
        metadata={"help": "Whether ot not to freeze the vision tower in MLLM training."},
    )
    freeze_multi_modal_projector: bool = field(
        default=True,
        metadata={"help": "Whether or not to freeze the multi modal projector in MLLM training."},
    )
    freeze_language_model: bool = field(
        default=False,
        metadata={"help": "Whether or not to freeze the language model in MLLM training."},
    )
    compute_accuracy: bool = field(
        default=False,
        metadata={"help": "Whether or not to compute the token-level accuracy at evaluation."},
    )
    disable_shuffling: bool = field(
        default=False,
        metadata={"help": "Whether or not to disable the shuffling of the training set."},
    )
    early_stopping_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Number of steps to stop training if the `metric_for_best_model` does not improve."},
    )
    plot_loss: bool = field(
        default=False,
        metadata={"help": "Whether or not to save the training loss curves."},
    )
    include_effective_tokens_per_second: bool = field(
        default=False,
        metadata={"help": "Whether or not to compute effective tokens per second."},
    )
    use_prompt_manager: bool = field(
        default=False,
        metadata={"help": "Enable multilingual soft-prompt manager."},
    )

    use_prompts: bool = field(
        default=False,
        metadata={"help": "Whether to allow training the prompt parameters."},
    )
    train_prompts: bool = field(
        default=False,
        metadata={"help": "Whether to allow training the prompt parameters."},
    )

    save_prompts: bool = field(
        default=False,
        metadata={"help": "Whether to save learned prompts during checkpoint saving."}
    )

    prompt_save_dir: str = field(
        default="prompts",
        metadata={"help": "Directory where prompts will be saved. Aliased with prompt_dir for compatibility."}
    )
    lang_pairs: Optional[list[str]] = field(
        default=None,
        metadata={"help": "List of language-pair strings, e.g., ['zh->en','en->my']."},
    )
    lang_map_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional path to save/load lang_pair mapping JSON file."},
    )

    encoder_save_steps: int = field(
        default=0,
        metadata={"help": "Save encoder checkpoints every N global steps to {prompt_save_dir}/encoder_check_{step}/. "
                          "Set to 0 to disable periodic encoder saving."}
    )
    prompt_encoder_type: Optional[str] = field(
        default="sda_ra",
        metadata={"help": "Type of prompt encoder. Only 'sda_ra' is supported. Set to 'none' to disable."}
    )
    prompt_encoder_prompt_length: int = field(
        default=32,
        metadata={"help": "Number of prompt tokens P to generate."}
    )
    prompt_encoder_dropout: float = field(
        default=0.1,
        metadata={"help": "Dropout rate for encoder cross-attention blocks."}
    )
    prompt_encoder_lr: Optional[float] = field(
        default=None,
        metadata={"help": "Learning rate for SDA-RA parameters. If None, falls back to global learning_rate."}
    )
    prompt_encoder_ffn_mult: float = field(
        default=2.0,
        metadata={"help": "FFN dimension multiplier for cross-attention blocks."}
    )
    prompt_encoder_output_scale: float = field(
        default=1.0,
        metadata={"help": "Initial value for learnable output scale factor s."}
    )
    prompt_encoder_num_lang_pairs: int = field(
        default=10,
        metadata={"help": "Total number of translation directions (N). Must equal len(lang_pairs) for the current run/checkpoint."}
    )

    sda_trunk_dim: int = field(
        default=256,
        metadata={"help": "Dimension of the shared cross-attention trunk."}
    )
    sda_num_adapters: int = field(
        default=4,
        metadata={"help": "Number of low-rank direction adapters K in the adapter pool."}
    )
    sda_adapter_rank: int = field(
        default=16,
        metadata={"help": "Rank of each low-rank direction adapter."}
    )
    sda_dir_dim: int = field(
        default=64,
        metadata={"help": "Dimension of direction embedding e_dir."}
    )
    sda_up_proj_init_std: float = field(
        default=5e-4,
        metadata={"help": "Std for near-zero initialization of W_up weight matrix. Slightly larger than the original setting to strengthen CE gradients reaching the private branch."}
    )
    sda_enable_adapters: bool = field(
        default=True,
        metadata={"help": "Ablation: disable direction adapter pool (trunk-only baseline)."}
    )
    sda_enable_contrastive_loss: bool = field(
        default=True,
        metadata={"help": "Ablation: disable contrastive routing loss (cosine similarity penalty)."}
    )
    sda_enable_balance_loss: bool = field(
        default=False,
        metadata={"help": "Ablation: disable expert balance loss (Switch Transformer style)."}
    )
    sda_enable_anchor_loss: bool = field(
        default=False,
        metadata={"help": "Enable soft direction-to-expert anchor loss for 1-direction-1-main-expert routing."}
    )
    sda_contrastive_loss_weight: float = field(
        default=0.1,
        metadata={"help": "Weight coefficient for contrastive routing loss."}
    )
    sda_balance_loss_weight: float = field(
        default=0.0,
        metadata={"help": "Weight coefficient for expert balance loss."}
    )
    sda_anchor_loss_weight_start: float = field(
        default=0.30,
        metadata={"help": "Initial weight of anchor loss before linear annealing."}
    )
    sda_anchor_loss_weight_end: float = field(
        default=0.05,
        metadata={"help": "Final weight of anchor loss after linear annealing."}
    )
    sda_anchor_margin: float = field(
        default=0.15,
        metadata={"help": "Margin used by anchor margin loss: z(anchor) should exceed all competitors by at least this value."}
    )
    sda_anchor_margin_weight: float = field(
        default=0.25,
        metadata={"help": "Internal coefficient applied to anchor margin loss branch."}
    )
    sda_anchor_target_main_prob: float = field(
        default=0.78,
        metadata={"help": "Soft target probability assigned to the main anchor expert."}
    )
    sda_anchor_target_overflow_prob: float = field(
        default=0.08,
        metadata={"help": "Soft target probability assigned to each overflow expert."}
    )
    sda_anchor_overflow_expert_ids: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated list or JSON array of overflow experts. If omitted and anchor loss is enabled, defaults to the unassigned tail experts."}
    )
    sda_anchor_expert_map: Optional[str] = field(
        default=None,
        metadata={"help": "JSON dict mapping lang_pair -> anchor expert id. If omitted and anchor loss is enabled, defaults to sequential 0..len(lang_pairs)-1."}
    )
    lang_loss_weights: Optional[str] = field(
        default=None,
        metadata={"help": "JSON string of per-direction loss weights, e.g. '{\"en-my\":3.0,\"my-en\":3.0}'. "
                          "Weights are batch-normalized: weighted_loss / sum(weights_in_batch)."}
    )
    vector_prompt_position: str = field(
        default="prefix",
        metadata={"help": "Where to inject prompt embeddings: 'prefix' or 'suffix'."}
    )
    enable_text_prefix: bool = field(
        default=True,
        metadata={"help": "Whether to prepend a textual translation instruction prefix to input_ids."}
    )
    text_prefix_template: str = field(
        default="Translate {src} into {tgt}.",
        metadata={"help": "Template for textual prefix. Placeholders: {src}, {tgt}, {lang_pair}."}
    )
    finetuning_max_grad_norm: Optional[float] = field(
        default=None,
        metadata={"help": "Override TrainingArguments.max_grad_norm. None = use Trainer default."}
    )

    current_lang_pair: Optional[str] = field(
        default=None,
        metadata={"help": "Runtime: the current language-pair being processed (e.g. 'zh-en'). Set by data pipeline."}
    )
    def __post_init__(self):
        if not self.use_prompt_manager:
            self.train_prompts = False
            self.save_prompts = False
            self.use_prompts = False
            if str(self.prompt_encoder_type or "").lower() == "sda_ra":
                self.prompt_encoder_type = "none"
        if str(self.prompt_encoder_type or "").lower() == "sda_ra":
            self.use_prompt_manager = True
            self.use_prompts = True
        self._parsed_lang_loss_weights: dict = {}
        if self.lang_loss_weights:
            try:
                import json
                self._parsed_lang_loss_weights = json.loads(self.lang_loss_weights)
            except Exception:
                pass

        def split_arg(arg):
            if isinstance(arg, str):
                return [item.strip() for item in arg.split(",")]
            return arg

        self.freeze_trainable_modules: list[str] = split_arg(self.freeze_trainable_modules)
        self.freeze_extra_modules: Optional[list[str]] = split_arg(self.freeze_extra_modules)
        self.lora_alpha: int = self.lora_alpha or self.lora_rank * 2
        self.lora_target: list[str] = split_arg(self.lora_target)
        self.oft_target: list[str] = split_arg(self.oft_target)
        self.additional_target: Optional[list[str]] = split_arg(self.additional_target)
        self.galore_target: list[str] = split_arg(self.galore_target)
        self.apollo_target: list[str] = split_arg(self.apollo_target)
        self.use_ref_model = self.stage == "dpo" and self.pref_loss not in ["orpo", "simpo"]
        self.lang_pairs = split_arg(self.lang_pairs)

        self._parsed_sda_anchor_expert_map: dict[str, int] = {}
        self._parsed_sda_anchor_overflow_expert_ids: list[int] = []
        self._parsed_sda_anchor_expert_by_lang_id: Optional[list[int]] = None

        def _parse_int_list(arg) -> list[int]:
            if arg is None:
                return []
            if isinstance(arg, list):
                return [int(x) for x in arg]
            if isinstance(arg, tuple):
                return [int(x) for x in arg]
            if isinstance(arg, str):
                text = arg.strip()
                if not text:
                    return []
                try:
                    import json
                    parsed = json.loads(text)
                    if isinstance(parsed, list):
                        return [int(x) for x in parsed]
                except Exception:
                    pass
                return [int(item.strip()) for item in text.split(",") if item.strip()]
            raise ValueError(f"Unsupported overflow expert ids format: {arg!r}")

        def _parse_anchor_map(arg) -> dict[str, int]:
            if arg is None:
                return {}
            if isinstance(arg, dict):
                return {str(k).strip(): int(v) for k, v in arg.items()}
            if isinstance(arg, str):
                text = arg.strip()
                if not text:
                    return {}
                try:
                    import json
                    parsed = json.loads(text)
                    if isinstance(parsed, dict):
                        return {str(k).strip(): int(v) for k, v in parsed.items()}
                except Exception as exc:
                    raise ValueError(f"Failed to parse sda_anchor_expert_map: {exc}") from exc
            raise ValueError(f"Unsupported sda_anchor_expert_map format: {arg!r}")

        self._parsed_sda_anchor_overflow_expert_ids = _parse_int_list(self.sda_anchor_overflow_expert_ids)
        self._parsed_sda_anchor_expert_map = _parse_anchor_map(self.sda_anchor_expert_map)

        if str(self.prompt_encoder_type or "").lower() == "sda_ra" and self.lang_pairs:
            expected = int(getattr(self, "prompt_encoder_num_lang_pairs", 0) or 0)
            actual = len(self.lang_pairs)
            if expected != actual:
                raise ValueError(
                    "配置不一致: prompt_encoder_num_lang_pairs=%d, 但 lang_pairs 有 %d 个方向。"
                    "\n这会导致 checkpoint 中出现未使用的 direction slots（如 dir_8/dir_9），"
                    "并在训练/推理/分析阶段造成方向 ID 错位。"
                    "\n请将 prompt_encoder_num_lang_pairs 改为 %d。" % (expected, actual, actual)
                )

            if self.sda_enable_anchor_loss:
                if int(self.sda_num_adapters) < actual:
                    raise ValueError(
                        "启用软一向一专时，sda_num_adapters 必须 >= len(lang_pairs)。"
                        f" 当前 sda_num_adapters={self.sda_num_adapters}, len(lang_pairs)={actual}。"
                    )

                if not self._parsed_sda_anchor_expert_map:
                    self._parsed_sda_anchor_expert_map = {
                        str(lp).strip(): idx for idx, lp in enumerate(self.lang_pairs)
                    }

                if not self._parsed_sda_anchor_overflow_expert_ids:
                    self._parsed_sda_anchor_overflow_expert_ids = list(range(actual, int(self.sda_num_adapters)))

                unknown = sorted(set(self._parsed_sda_anchor_expert_map.keys()) - set(self.lang_pairs))
                if unknown:
                    raise ValueError(f"sda_anchor_expert_map 包含未知方向: {unknown}")

                missing = [lp for lp in self.lang_pairs if lp not in self._parsed_sda_anchor_expert_map]
                if missing:
                    raise ValueError(f"sda_anchor_expert_map 缺少方向: {missing}")

                anchor_ids = [int(self._parsed_sda_anchor_expert_map[lp]) for lp in self.lang_pairs]
                if len(set(anchor_ids)) != len(anchor_ids):
                    raise ValueError(
                        f"sda_anchor_expert_map 必须为每个方向分配唯一主专家，当前存在重复: {anchor_ids}"
                    )

                if any(idx < 0 or idx >= int(self.sda_num_adapters) for idx in anchor_ids):
                    raise ValueError(
                        f"sda_anchor_expert_map 中存在越界 expert id，合法范围是 [0, {int(self.sda_num_adapters) - 1}]"
                    )

                if any(idx < 0 or idx >= int(self.sda_num_adapters) for idx in self._parsed_sda_anchor_overflow_expert_ids):
                    raise ValueError(
                        f"sda_anchor_overflow_expert_ids 中存在越界 expert id，合法范围是 [0, {int(self.sda_num_adapters) - 1}]"
                    )

                overlap = sorted(set(anchor_ids) & set(self._parsed_sda_anchor_overflow_expert_ids))
                if overlap:
                    raise ValueError(f"overflow experts 不能与 anchor experts 重叠: {overlap}")

                total_target_mass = float(self.sda_anchor_target_main_prob) + len(self._parsed_sda_anchor_overflow_expert_ids) * float(self.sda_anchor_target_overflow_prob)
                if total_target_mass > 1.0 + 1e-8:
                    raise ValueError(
                        "anchor soft target 概率和超过 1.0，请检查 main/overflow 设定。"
                        f" 当前 total={total_target_mass:.6f}"
                    )

                self._parsed_sda_anchor_expert_by_lang_id = anchor_ids

        assert self.finetuning_type in ["lora", "oft", "freeze", "full"], "Invalid fine-tuning method."
        assert self.ref_model_quantization_bit in [None, 8, 4], "We only accept 4-bit or 8-bit quantization."
        assert self.reward_model_quantization_bit in [None, 8, 4], "We only accept 4-bit or 8-bit quantization."

        if self.stage == "ppo" and self.reward_model is None:
            raise ValueError("`reward_model` is necessary for PPO training.")

        if self.stage == "ppo" and self.reward_model_type == "lora" and self.finetuning_type != "lora":
            raise ValueError("`reward_model_type` cannot be lora for Freeze/Full PPO training.")

        if self.stage == "ppo" and self.reward_model_type == "oft" and self.finetuning_type != "oft":
            raise ValueError("`reward_model_type` cannot be oft for Freeze/Full PPO training.")

        if self.stage == "dpo" and self.pref_loss != "sigmoid" and self.dpo_label_smoothing > 1e-6:
            raise ValueError("`dpo_label_smoothing` is only valid for sigmoid loss function.")

        if self.use_llama_pro and self.finetuning_type == "full":
            raise ValueError("`use_llama_pro` is only valid for Freeze or LoRA training.")

        if self.finetuning_type == "lora" and (self.use_galore or self.use_apollo or self.use_badam):
            raise ValueError("Cannot use LoRA with GaLore, APOLLO or BAdam together.")

        if int(self.use_galore) + int(self.use_apollo) + (self.use_badam) > 1:
            raise ValueError("Cannot use GaLore, APOLLO or BAdam together.")

        if self.pissa_init and (self.stage in ["ppo", "kto"] or self.use_ref_model):
            raise ValueError("Cannot use PiSSA for current training stage.")

        if self.finetuning_type != "lora":
            if self.loraplus_lr_ratio is not None:
                raise ValueError("`loraplus_lr_ratio` is only valid for LoRA training.")

            if self.use_rslora:
                raise ValueError("`use_rslora` is only valid for LoRA training.")

            if self.use_dora:
                raise ValueError("`use_dora` is only valid for LoRA training.")

            if self.pissa_init:
                raise ValueError("`pissa_init` is only valid for LoRA training.")

    def to_dict(self) -> dict[str, Any]:
        args = asdict(self)
        args = {k: f"<{k.upper()}>" if k.endswith("api_key") else v for k, v in args.items()}
        return args
