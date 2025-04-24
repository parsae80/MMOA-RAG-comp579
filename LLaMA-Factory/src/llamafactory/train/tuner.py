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
import shutil
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.distributed as dist
from transformers import PreTrainedModel

from ..data import get_template_and_fix_tokenizer
from ..extras import logging
from ..extras.constants import V_HEAD_SAFE_WEIGHTS_NAME, V_HEAD_WEIGHTS_NAME
from ..extras.misc import infer_optim_dtype
from ..extras.packages import is_ray_available
from ..hparams import get_infer_args, get_ray_args, get_train_args, read_args
from ..model import load_model, load_tokenizer
from .callbacks import LogCallback, PissaConvertCallback, ReporterCallback
from .dpo import run_dpo
from .kto import run_kto
from .ppo import run_ppo
from .pt import run_pt
from .rm import run_rm
from .sft import run_sft
from .trainer_utils import get_ray_trainer, get_swanlab_callback

if is_ray_available():
    import ray
    from ray.train.huggingface.transformers import RayTrainReportCallback

if TYPE_CHECKING:
    from transformers import TrainerCallback


logger = logging.get_logger(__name__)


def _training_function(config: dict[str, Any]) -> None:
    args = config.get("args")
    callbacks: list[Any] = config.get("callbacks")
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)

    # always add logging/reporting callbacks
    callbacks.append(LogCallback())
    if finetuning_args.pissa_convert:
        callbacks.append(PissaConvertCallback())
    if finetuning_args.use_swanlab:
        callbacks.append(get_swanlab_callback(finetuning_args))
    callbacks.append(ReporterCallback(model_args, data_args, finetuning_args, generating_args))

    # dispatch by stage
    if finetuning_args.stage == "pt":
        run_pt(model_args, data_args, training_args, finetuning_args, callbacks)
    elif finetuning_args.stage == "sft":
        run_sft(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)
    elif finetuning_args.stage == "rm":
        run_rm(model_args, data_args, training_args, finetuning_args, callbacks)
    # 🔥 UPDATED: handle both new PPO modes
    elif finetuning_args.stage in ("vanilla_ppo", "finance_ppo"):
        run_ppo(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)
    elif finetuning_args.stage == "dpo":
        run_dpo(model_args, data_args, training_args, finetuning_args, callbacks)
    elif finetuning_args.stage == "kto":
        run_kto(model_args, data_args, training_args, finetuning_args, callbacks)
    else:
        raise ValueError(f"Unknown task: {finetuning_args.stage}.")

    # clean up distributed if not using Ray
    if is_ray_available() and ray.is_initialized():
        return
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception as e:
        logger.warning(f"Failed to destroy process group: {e}.")


def run_exp(args: Optional[dict[str, Any]] = None, callbacks: Optional[list["TrainerCallback"]] = None) -> None:
    """
    Entry point for `llamafactory-cli train`.
    Reads args, optionally spins up a Ray trainer, or just calls _training_function.
    """
    args = read_args(args)
    if "-h" in args or "--help" in args:
        get_train_args(args)

    ray_args = get_ray_args(args)
    callbacks = callbacks or []

    if ray_args.use_ray:
        callbacks.append(RayTrainReportCallback())
        trainer = get_ray_trainer(
            training_function=_training_function,
            train_loop_config={"args": args, "callbacks": callbacks},
            ray_args=ray_args,
        )
        trainer.fit()
    else:
        _training_function(config={"args": args, "callbacks": callbacks})


def export_model(args: Optional[dict[str, Any]] = None) -> None:
    """
    Export a trained model for inference.
    """
    model_args, data_args, finetuning_args, _ = get_infer_args(args)

    if model_args.export_dir is None:
        raise ValueError("Please specify `export_dir` to save model.")

    if model_args.adapter_name_or_path is not None and model_args.export_quantization_bit is not None:
        raise ValueError("Please merge adapters before quantizing the model.")

    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    processor = tokenizer_module["processor"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    model = load_model(tokenizer, model_args, finetuning_args)

    # adjust dtype for export
    if getattr(model, "quantization_method", None) is not None:
        setattr(model.config, "torch_dtype", torch.float16)
    else:
        output_dtype = (
            infer_optim_dtype(torch.bfloat16)
            if model_args.infer_dtype == "auto" and model.config.torch_dtype == torch.float32
            else getattr(torch, model_args.infer_dtype or model.config.torch_dtype)
        )
        setattr(model.config, "torch_dtype", output_dtype)
        model = model.to(output_dtype)
        logger.info_rank0(f"Convert model dtype to: {output_dtype}.")

    # save
    model.save_pretrained(
        save_directory=model_args.export_dir,
        max_shard_size=f"{model_args.export_size}GB",
        safe_serialization=(not model_args.export_legacy_format),
    )
    if model_args.export_hub_model_id:
        model.push_to_hub(
            model_args.export_hub_model_id,
            token=model_args.hf_hub_token,
            max_shard_size=f"{model_args.export_size}GB",
            safe_serialization=(not model_args.export_legacy_format),
        )

    # copy over value‐head if present
    vhead_src = (
        os.path.join(model_args.adapter_name_or_path[-1], V_HEAD_SAFE_WEIGHTS_NAME)
        if model_args.adapter_name_or_path
        else os.path.join(model_args.model_name_or_path, V_HEAD_SAFE_WEIGHTS_NAME)
    )
    if os.path.exists(vhead_src):
        shutil.copy(vhead_src, os.path.join(model_args.export_dir, V_HEAD_SAFE_WEIGHTS_NAME))
        logger.info_rank0(f"Copied valuehead to {model_args.export_dir}.")

    # finally save tokenizer + processor
    try:
        tokenizer.padding_side = "left"
        tokenizer.save_pretrained(model_args.export_dir)
        if processor:
            processor.save_pretrained(model_args.export_dir)
    except Exception as e:
        logger.warning_rank0(f"Cannot save tokenizer/processor: {e}.")
