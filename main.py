from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL_DIR = "FLUX.1-dev-penguinoh-generator"
DEFAULT_IMAGES_DIR = "./data/images"
USE_WANDB = True
WANDB_ENTITY = "eigenben-rubidiom"
WANDB_PROJECT = "penguinoh-generator"

@dataclass
class SharedConfig:
    instance_name: str = "pngnh"
    class_name: str = "Stuffed Penguin"
    model_name: str = "black-forest-labs/FLUX.1-dev"

@dataclass
class TrainConfig(SharedConfig):
    prefix: str = "a photo of"
    postfix: str = ""
    resolution: int = 512
    train_batch_size: int = 1
    rank: int = 16  # lora rank
    gradient_accumulation_steps: int = 1
    learning_rate: float = 4e-4
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0
    max_train_steps: int = 500
    checkpointing_steps: int = 1000
    seed: int = 117

def download(model_dir, config):
    import torch
    from diffusers import DiffusionPipeline
    from huggingface_hub import snapshot_download

    snapshot_download(
        config.model_name,
        local_dir=model_dir,
        ignore_patterns=["*.pt", "*.bin"],  # using safetensors
    )

    DiffusionPipeline.from_pretrained(model_dir, torch_dtype=torch.bfloat16)

def train(model_dir, images_dir, config):
    import torch
    import subprocess
    from accelerate.utils import write_basic_config

    if torch.cuda.is_available():
        write_basic_config(mixed_precision="bf16")  # set up hugging face accelerate library for fast training
    else:
        write_basic_config(mixed_precision="no")

    instance_phrase = f"{config.instance_name} the {config.class_name}"
    prompt = f"{config.prefix} {instance_phrase} {config.postfix}".strip()

    # The model training is packaged as a script, so we have to execute it as a subprocess, which adds some boilerplate
    def _exec_subprocess(cmd: list[str]):
        """Executes subprocess and prints log to terminal while subprocess is running."""
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        with process.stdout as pipe:
            for line in iter(pipe.readline, b""):
                line_str = line.decode()
                print(f"{line_str}", end="")

        if exitcode := process.wait() != 0:
            raise subprocess.CalledProcessError(exitcode, "\n".join(cmd))

    # run training -- see huggingface accelerate docs for details
    print("Launching dreambooth training script...")
    _exec_subprocess(
        [
            "accelerate",
            "launch",
            "dreambooth/train_dreambooth_lora_flux.py",
            ("--mixed_precision=bf16" if torch.cuda.is_available() else "--mixed_precision=no"),  # half-precision floats most of the time for faster training
            f"--pretrained_model_name_or_path={model_dir}",
            f"--instance_data_dir={images_dir}",
            f"--output_dir={model_dir}",
            f"--instance_prompt={prompt}",
            f"--resolution={config.resolution}",
            f"--train_batch_size={config.train_batch_size}",
            f"--gradient_accumulation_steps={config.gradient_accumulation_steps}",
            f"--learning_rate={config.learning_rate}",
            f"--lr_scheduler={config.lr_scheduler}",
            f"--lr_warmup_steps={config.lr_warmup_steps}",
            f"--max_train_steps={config.max_train_steps}",
            f"--checkpointing_steps={config.checkpointing_steps}",
            f"--seed={config.seed}",  # increased reproducibility by seeding the RNG
            f"--push_to_hub",
        ]
        + (
            [
                "--report_to=wandb",
                # validation output tracking is useful, but currently broken for Flux LoRA training
                # f"--validation_prompt={prompt} in space",  # simple test prompt
                # f"--validation_epochs={config.max_train_steps // 5}",
            ]
            if USE_WANDB
            else []
        ),
    )

if __name__ == "__main__":
    import argparse
    import os

    if USE_WANDB:
        os.environ["WANDB_ENTITY"] = WANDB_ENTITY
        os.environ["WANDB_PROJECT"] = WANDB_PROJECT

    parser = argparse.ArgumentParser(description="Download model assets and/or run training.")
    parser.add_argument(
        "action",
        nargs="?",
        choices=["download", "train"],
        help='Choose "download" to only fetch model assets or "train" to only run training. Omit to do both.',
    )
    args = parser.parse_args()

    train_config = TrainConfig()

    if args.action in (None, "download"):
        download(DEFAULT_MODEL_DIR, train_config)
    if args.action in (None, "train"):
        train(DEFAULT_MODEL_DIR, DEFAULT_IMAGES_DIR, train_config)
