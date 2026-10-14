"""Settings shared by dataset prep, training, generation and the Modal app."""

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = REPO_ROOT / "data" / "train"
RUNS_DIR = REPO_ROOT / "runs"
OUTPUT_DIR = REPO_ROOT / "output"
TRAIN_SCRIPT = REPO_ROOT / "dreambooth" / "train_dreambooth_lora_flux2_klein.py"

INSTANCE_NAME = "pngnh"
CLASS_NAME = "Stuffed Penguin"
TRIGGER = f"{INSTANCE_NAME} the {CLASS_NAME}"
INSTANCE_PROMPT = f"a photo of {TRIGGER}"

BASE_MODEL = "black-forest-labs/FLUX.2-klein-base-4B"
DISTILLED_MODEL = "black-forest-labs/FLUX.2-klein-4B"
HUB_REPO = "eigenben/FLUX.2-klein-base-4B-penguinoh"
LORA_WEIGHT_NAME = "pytorch_lora_weights.safetensors"

WANDB_ENTITY = "eigenben-rubidiom"


@dataclass
class TrainConfig:
    base_model: str = BASE_MODEL
    resolution: int = 1024  # aspect-ratio buckets are derived from this
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    rank: int = 32  # lora_alpha is set to match, so the effective scale is 1.0
    learning_rate: float = 1e-4
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 100
    max_train_steps: int = 2000  # ~38 epochs over 52 images
    checkpointing_steps: int = 250
    validation_prompt: str = f"a watercolor painting of {TRIGGER} surfing a big wave"
    validation_epochs: int = 10
    seed: int = 117
    report_to_wandb: bool = True


@dataclass(frozen=True)
class Sampler:
    model: str
    steps: int
    guidance: float


# Settings from each model card. The LoRA is trained on base; distilled is the fast option.
SAMPLERS = {
    "base": Sampler(BASE_MODEL, steps=50, guidance=4.0),
    "distilled": Sampler(DISTILLED_MODEL, steps=4, guidance=1.0),
}
