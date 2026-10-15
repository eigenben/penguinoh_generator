"""Train a LoRA by running the vendored diffusers DreamBooth script under accelerate."""

import os
import subprocess
import sys
from pathlib import Path

from penguinoh.config import (
    HUB_REPO,
    INSTANCE_PROMPT,
    LORA_WEIGHT_NAME,
    MODEL_CARD,
    SAMPLES_DIR,
    TRAIN_DIR,
    TRAIN_SCRIPT,
    WANDB_ENTITY,
    TrainConfig,
)


def build_command(config: TrainConfig, output_dir: Path, train_dir: Path, mixed_precision: str) -> list[str]:
    cmd = [
        sys.executable, "-m", "accelerate.commands.launch",
        "--num_processes=1",
        "--num_machines=1",
        "--dynamo_backend=no",
        f"--mixed_precision={mixed_precision}",
        str(TRAIN_SCRIPT),
        f"--pretrained_model_name_or_path={config.base_model}",
        f"--dataset_name={train_dir}",
        "--image_column=image",
        "--caption_column=text",
        f"--instance_prompt={INSTANCE_PROMPT}",  # required by the script; per-image captions take precedence
        f"--output_dir={output_dir}",
        f"--mixed_precision={mixed_precision}",
        f"--resolution={config.resolution}",
        "--use_aspect_ratio_buckets",
        "--cache_latents",
        "--gradient_checkpointing",
        f"--train_batch_size={config.train_batch_size}",
        f"--gradient_accumulation_steps={config.gradient_accumulation_steps}",
        f"--rank={config.rank}",
        f"--lora_alpha={config.rank}",
        f"--learning_rate={config.learning_rate}",
        f"--lr_scheduler={config.lr_scheduler}",
        f"--lr_warmup_steps={config.lr_warmup_steps}",
        f"--max_train_steps={config.max_train_steps}",
        f"--checkpointing_steps={config.checkpointing_steps}",
        f"--seed={config.seed}",
    ]
    if config.report_to_wandb:
        cmd += [
            "--report_to=wandb",
            f"--validation_prompt={config.validation_prompt}",
            f"--validation_epochs={config.validation_epochs}",
            "--num_validation_images=2",
        ]
    else:
        cmd += ["--report_to=tensorboard", "--skip_final_inference"]
    return cmd


def train(config: TrainConfig, output_dir: Path, train_dir: Path = TRAIN_DIR):
    import torch

    mixed_precision = "bf16" if torch.cuda.is_available() else "no"
    env = {**os.environ, "WANDB_ENTITY": os.environ.get("WANDB_ENTITY", WANDB_ENTITY)}
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training {config.base_model} LoRA into {output_dir}", flush=True)
    subprocess.run(build_command(config, output_dir, train_dir, mixed_precision), env=env, check=True)


def upload(output_dir: Path, repo_id: str = HUB_REPO):
    """Push the final LoRA and each checkpoint's LoRA weights (not optimizer state), plus our model card."""
    from huggingface_hub import CommitOperationAdd, create_commit, create_repo

    weights = [output_dir / LORA_WEIGHT_NAME, *sorted(output_dir.glob(f"checkpoint-*/{LORA_WEIGHT_NAME}"))]
    operations = [CommitOperationAdd(str(path.relative_to(output_dir)), path) for path in weights if path.exists()]
    if MODEL_CARD.exists():
        operations.append(CommitOperationAdd("README.md", MODEL_CARD))
        operations += [CommitOperationAdd(f"samples/{p.name}", p) for p in sorted(SAMPLES_DIR.glob("*.jpg"))]

    create_repo(repo_id, exist_ok=True)
    create_commit(repo_id, operations, commit_message=f"Upload {output_dir.name}")
    print(f"Uploaded {len(operations)} files from {output_dir} to https://huggingface.co/{repo_id}")
