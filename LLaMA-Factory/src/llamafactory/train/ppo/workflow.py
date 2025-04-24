# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's TRL library.
# https://github.com/huggingface/trl/blob/v0.8.0/examples/scripts/ppo.py
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

from typing import TYPE_CHECKING, Optional

from ...data import MultiModalDataCollatorForSeq2Seq, get_dataset, get_template_and_fix_tokenizer
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..callbacks import fix_valuehead_checkpoint
from ..trainer_utils import create_ref_model, create_reward_model
from .trainer import CustomPPOTrainer, VanillaPPOTrainer, FinancePPOTrainer

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


def run_ppo(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    # 1) Load tokenizer & template
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)

    # 2) Always pull in a 'ppo' dataset – this gives `train_dataset` and an `eval_dataset`
    dataset_module = get_dataset(
        template=template,
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        stage="ppo",  # <-- force ppo here
        **tokenizer_module,
    )
    # PPOTrainer cannot accept eval_dataset: drop it
    dataset_module.pop("eval_dataset", None)

    # 3) Load model + value head
    model = load_model(
        tokenizer,
        model_args,
        finetuning_args,
        training_args.do_train,
        add_valuehead=True,
    )

    # 4) Build data collator
    tokenizer.padding_side = "left"
    data_collator = MultiModalDataCollatorForSeq2Seq(
        template=template,
        model=model,
        **tokenizer_module,
    )

    # 5) Reference model
    ref_model = create_ref_model(model_args, finetuning_args, add_valuehead=True)

    # 6) Only the original 'ppo' needs a HuggingFace reward_model adapter
    if finetuning_args.stage == "ppo":
        reward_model = create_reward_model(model, model_args, finetuning_args)
    else:
        reward_model = None

    # 7) Choose trainer class by stage name
    trainer_cls = {
        "vanilla_ppo": VanillaPPOTrainer,
        "finance_ppo": FinancePPOTrainer,
    }.get(finetuning_args.stage, CustomPPOTrainer)

    # 8) Instantiate
    ppo_trainer = trainer_cls(
        model_args=model_args,
        training_args=training_args,
        finetuning_args=finetuning_args,
        generating_args=generating_args,
        callbacks=callbacks,
        model=model,
        reward_model=reward_model,
        ref_model=ref_model,
        data_collator=data_collator,
        **dataset_module,    # only train_dataset now
        **tokenizer_module,
    )

    # 9) Train
    if training_args.do_train:
        ppo_trainer.ppo_train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        ppo_trainer.save_model()
        if training_args.should_save:
            fix_valuehead_checkpoint(model, training_args.output_dir, training_args.save_safetensors)
        ppo_trainer.save_state()
        if ppo_trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "reward"])
    