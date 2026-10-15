"""Train and run the Penguinoh LoRA on Modal GPUs.

    modal run --detach modal_app.py::train                  # train (and optionally --push) into the runs volume
    modal run modal_app.py::generate --run <run-name>       # compare every checkpoint of a run on prompts/eval.txt
    modal run modal_app.py::upload --run <run-name> --root-checkpoint checkpoint-N   # publish to the Hub
    modal run modal_app.py::generate --prompt "surfing in Hawaii" --seeds 1,2,3
    modal deploy modal_app.py                               # deploy Generator for other apps to call
"""

import io
from pathlib import Path

import modal

from penguinoh.config import DEFAULT_LORA, HUB_REPO, LORA_WEIGHT_NAME, OUTPUT_DIR, SAMPLERS, TrainConfig

REMOTE_ROOT = Path("/root")  # where Modal puts local Python source, so penguinoh.config.REPO_ROOT resolves here
HF_CACHE = "/cache/huggingface"
GPU = "L40S"

app = modal.App(name="penguinoh-generator")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync()
    .env({"HF_HOME": HF_CACHE, "HF_XET_HIGH_PERFORMANCE": "1"})
    .add_local_dir("dreambooth", str(REMOTE_ROOT / "dreambooth"))
    .add_local_dir("data/train", str(REMOTE_ROOT / "data" / "train"))
    .add_local_dir("docs", str(REMOTE_ROOT / "docs"))
    .add_local_python_source("penguinoh")
)

hf_cache = modal.Volume.from_name("penguinoh-generator-hf-cache", create_if_missing=True)
runs_volume = modal.Volume.from_name("penguinoh-generator-runs", create_if_missing=True)
volumes = {HF_CACHE: hf_cache, str(REMOTE_ROOT / "runs"): runs_volume}

huggingface_secret = modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])
wandb_secret = modal.Secret.from_name("wandb-secret", required_keys=["WANDB_API_KEY"])


@app.function(image=image, gpu=GPU, volumes=volumes, secrets=[huggingface_secret, wandb_secret], timeout=4 * 60 * 60)
def train_remote(run_name: str, config: TrainConfig, push: bool):
    from penguinoh.config import RUNS_DIR
    from penguinoh.train import train, upload

    output_dir = RUNS_DIR / run_name
    try:
        train(config, output_dir)
    finally:
        runs_volume.commit()  # keep whatever checkpoints exist even if training fails
    if push:
        upload(output_dir)


@app.function(image=image, volumes=volumes, secrets=[huggingface_secret], timeout=30 * 60)
def upload_remote(run_name: str, repo_id: str, root_checkpoint: str | None):
    from penguinoh.config import RUNS_DIR
    from penguinoh.train import upload

    upload(RUNS_DIR / run_name, repo_id, root_checkpoint)


@app.cls(image=image, gpu=GPU, volumes=volumes, secrets=[huggingface_secret], scaledown_window=300, timeout=60 * 60,
         max_containers=4)
class Generator:
    sampler: str = modal.parameter(default="base")

    @modal.enter()
    def load(self):
        from penguinoh.generate import load_pipeline

        self.pipe = load_pipeline(self.sampler)

    @modal.method()
    def generate(self, lora: str, prompts: list[str], seeds: list[int], lora_scale: float = 1.0) -> list[bytes]:
        """Render every (prompt, seed) pair with one LoRA; returns PNG bytes in prompt-major order."""
        from penguinoh.generate import generate, parse_lora, set_lora

        runs_volume.reload()
        set_lora(self.pipe, parse_lora(lora), lora_scale)
        images = []
        for prompt in prompts:
            for seed in seeds:
                buffer = io.BytesIO()
                generate(self.pipe, prompt, seed, self.sampler).save(buffer, format="PNG")
                images.append(buffer.getvalue())
        return images


def run_loras(run_name: str) -> list[str]:
    """Every checkpoint of a run in the runs volume, in step order, then the final weights."""
    entries = {Path(entry.path).name for entry in runs_volume.listdir(run_name)}
    if not entries:
        raise ValueError(f"No run named {run_name} in the runs volume")
    checkpoints = sorted((e for e in entries if e.startswith("checkpoint-")), key=lambda e: int(e.split("-")[1]))
    final = [f"runs/{run_name}"] if LORA_WEIGHT_NAME in entries else []
    return [f"runs/{run_name}/{c}" for c in checkpoints] + final


@app.local_entrypoint()
def train(
    run_name: str = "",
    max_train_steps: int = TrainConfig.max_train_steps,
    checkpointing_steps: int = TrainConfig.checkpointing_steps,
    rank: int = TrainConfig.rank,
    learning_rate: float = TrainConfig.learning_rate,
    push: bool = False,
):
    from penguinoh.cli import default_run_name

    run_name = run_name or default_run_name()
    config = TrainConfig(
        max_train_steps=max_train_steps, checkpointing_steps=checkpointing_steps, rank=rank, learning_rate=learning_rate
    )
    print(f"Training run {run_name}: {config}")
    train_remote.remote(run_name, config, push)
    print(f"Finished {run_name}. Compare checkpoints with: modal run modal_app.py::generate --run {run_name}")


@app.local_entrypoint()
def generate(
    lora: str = DEFAULT_LORA,  # comma-separated: none, runs/<run>[/checkpoint-N], or owner/repo[/checkpoint-N]
    run: str = "",  # shorthand for every checkpoint of a run in the runs volume
    prompt: str = "",  # a single prompt instead of the prompts file
    prompts_file: str = "prompts/eval.txt",
    seeds: str = "1",
    sampler: str = "base",
    lora_scale: float = 1.0,
    out: str = str(OUTPUT_DIR),
):
    from datetime import datetime

    from PIL import Image

    from penguinoh.generate import parse_lora, read_prompts, save_batch

    if sampler not in SAMPLERS:
        raise ValueError(f"sampler must be one of {list(SAMPLERS)}")
    specs = run_loras(run) if run else lora.split(",")
    prompts = [prompt] if prompt else read_prompts(Path(prompts_file))
    seed_list = [int(s) for s in seeds.split(",")]

    print(f"Rendering {len(prompts) * len(seed_list)} images for each of {len(specs)} LoRAs")
    generator = Generator(sampler=sampler)
    batches = list(generator.generate.starmap([(spec, prompts, seed_list, lora_scale) for spec in specs]))
    results = {}
    for spec, pngs in zip(specs, batches):
        source = parse_lora(spec)
        results[source.label if source else "no LoRA"] = [Image.open(io.BytesIO(png)) for png in pngs]
    save_batch(results, prompts, seed_list, Path(out) / datetime.now().strftime("%Y%m%d-%H%M%S"))


@app.local_entrypoint()
def upload(run: str, root_checkpoint: str = "", hub_repo: str = HUB_REPO):
    upload_remote.remote(run, hub_repo, root_checkpoint or None)
