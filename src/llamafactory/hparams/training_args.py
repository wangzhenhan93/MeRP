import json

# --- compatibility shims for types referenced inside HF/transformers dataclasses ---
# Try to import likely modules for common types used in HF dataclasses.
# If not found, define a minimal stub so typing.get_type_hints() won't raise NameError.
def _attempt_import(target_name, candidates):
    import importlib
    for mod in candidates:
        try:
            m = importlib.import_module(mod)
            if hasattr(m, target_name):
                return getattr(m, target_name)
        except Exception:
            continue
    return None

_compat_map = {
    "ParallelismConfig": [
        "transformers.trainer_utils",
        "transformers.utils",
        "accelerate.state",
        "transformers"
    ],
    "DeepSpeedConfig": [
        "transformers.deepspeed",
        "deepspeed"
    ],
    "DeepspeedConfig": [
        "transformers.deepspeed",
        "deepspeed"
    ],
    "FSDPConfig": [
        "accelerate.state",
        "transformers.trainer_utils",
        "transformers.utils"
    ],
    # additional commonly referenced names (safe to stub)
    "ShardedTensor": ["torch.distributed._sharded_tensor", "torch.distributed"],
    "FullyShardedDataParallel": ["torch.distributed.fsdp", "torch.distributed"],
}

for _name, _mods in _compat_map.items():
    _obj = _attempt_import(_name, _mods)
    if _obj is not None:
        globals()[_name] = _obj
    else:
        if _name not in globals():
            class _Stub:  # type: ignore
                """Compatibility stub inserted to allow type-hint resolution."""
                pass
            _Stub.__name__ = _name
            globals()[_name] = _Stub
# --- end compatibility shims ---


# Ensure ParallelismConfig name is present so typing.get_type_hints() used by HfArgumentParser
# can resolve annotations that reference it. Try importing the real class from likely locations,
# otherwise provide a harmless stub.
try:
    from transformers.trainer_utils import ParallelismConfig
except Exception:
    try:
        from transformers.utils import ParallelismConfig
    except Exception:
        try:
            from accelerate.state import ParallelismConfig
        except Exception:
            class ParallelismConfig:  # type: ignore
                """Fallback stub for ParallelismConfig to allow dataclass type-hint resolution."""
                pass

from dataclasses import dataclass, field
from typing import Literal, Optional, Union

from transformers import Seq2SeqTrainingArguments
from transformers.training_args import _convert_str_dict

from ..extras.misc import use_ray


@dataclass
class RayArguments:
    r"""Arguments pertaining to the Ray training."""

    ray_run_name: Optional[str] = field(
        default=None,
        metadata={"help": "The training results will be saved at `<ray_storage_path>/ray_run_name`."},
    )
    ray_storage_path: str = field(
        default="./saves",
        metadata={"help": "The storage path to save training results to"},
    )
    ray_storage_filesystem: Optional[Literal["s3", "gs", "gcs"]] = field(
        default=None,
        metadata={"help": "The storage filesystem to use. If None specified, local filesystem will be used."},
    )
    ray_num_workers: int = field(
        default=1,
        metadata={"help": "The number of workers for Ray training. Default is 1 worker."},
    )
    resources_per_worker: Union[dict, str] = field(
        default_factory=lambda: {"GPU": 1},
        metadata={"help": "The resources per worker for Ray training. Default is to use 1 GPU per worker."},
    )
    placement_strategy: Literal["SPREAD", "PACK", "STRICT_SPREAD", "STRICT_PACK"] = field(
        default="PACK",
        metadata={"help": "The placement strategy for Ray training. Default is PACK."},
    )
    ray_init_kwargs: Optional[Union[dict, str]] = field(
        default=None,
        metadata={"help": "The arguments to pass to ray.init for Ray training. Default is None."},
    )

    def __post_init__(self):
        self.use_ray = use_ray()
        if isinstance(self.resources_per_worker, str) and self.resources_per_worker.startswith("{"):
            self.resources_per_worker = _convert_str_dict(json.loads(self.resources_per_worker))

        if isinstance(self.ray_init_kwargs, str) and self.ray_init_kwargs.startswith("{"):
            self.ray_init_kwargs = _convert_str_dict(json.loads(self.ray_init_kwargs))

        if self.ray_storage_filesystem is not None:
            if self.ray_storage_filesystem not in ["s3", "gs", "gcs"]:
                raise ValueError(
                    f"ray_storage_filesystem must be one of ['s3', 'gs', 'gcs'], got {self.ray_storage_filesystem}."
                )

            import pyarrow.fs as fs

            if self.ray_storage_filesystem == "s3":
                self.ray_storage_filesystem = fs.S3FileSystem()
            elif self.ray_storage_filesystem == "gs" or self.ray_storage_filesystem == "gcs":
                self.ray_storage_filesystem = fs.GcsFileSystem()


@dataclass
class TrainingArguments(RayArguments, Seq2SeqTrainingArguments):
    r"""Arguments pertaining to the trainer."""

    def __post_init__(self):
        Seq2SeqTrainingArguments.__post_init__(self)
        RayArguments.__post_init__(self)