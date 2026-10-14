"""Generate images with FLUX.2 [klein] plus a Penguinoh LoRA."""

import re
import textwrap
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from penguinoh.config import INSTANCE_PROMPT, LORA_WEIGHT_NAME, REPO_ROOT, SAMPLERS, TRIGGER

ADAPTER_NAME = "penguinoh"


@dataclass(frozen=True)
class LoraSource:
    """Where to load LoRA weights from: a local directory, or a Hub repo plus optional subfolder."""

    location: str
    subfolder: str | None = None

    @property
    def label(self) -> str:
        """Short name for filenames and grid headers, e.g. `klein4b-20261014-1200/checkpoint-500`."""
        if Path(self.location).is_absolute():
            return "/".join(Path(self.location).parts[-2:])
        return f"{self.location.split('/')[-1]}/{self.subfolder or 'final'}"


def parse_lora(spec: str) -> LoraSource | None:
    """`none`, a local run or checkpoint directory, or `owner/repo[/checkpoint-N]` on the Hub."""
    if spec == "none":
        return None
    for candidate in (Path(spec), REPO_ROOT / spec):
        if candidate.is_dir():
            return LoraSource(str(candidate.resolve()))
    owner, repo, *rest = spec.split("/")
    return LoraSource(f"{owner}/{repo}", "/".join(rest) or None)


def expand_prompt(text: str) -> str:
    """Prompts that don't mention Penguinoh are treated as a scene for a photo of him."""
    text = text.strip()
    return text if TRIGGER in text else f"{INSTANCE_PROMPT} {text}"


def device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_pipeline(sampler: str = "base"):
    import torch
    from diffusers import Flux2KleinPipeline

    pipe = Flux2KleinPipeline.from_pretrained(SAMPLERS[sampler].model, torch_dtype=torch.bfloat16)
    return pipe.to(device())


def set_lora(pipe, lora: LoraSource | None, scale: float = 1.0):
    if ADAPTER_NAME in pipe.get_list_adapters().get("transformer", []):
        pipe.delete_adapters(ADAPTER_NAME)
    if lora is None:
        return
    pipe.load_lora_weights(
        lora.location, subfolder=lora.subfolder, weight_name=LORA_WEIGHT_NAME, adapter_name=ADAPTER_NAME
    )
    pipe.set_adapters([ADAPTER_NAME], [scale])


def generate(pipe, prompt: str, seed: int, sampler: str = "base", steps: int | None = None,
             guidance: float | None = None, width: int = 1024, height: int = 1024) -> Image.Image:
    import torch

    settings = SAMPLERS[sampler]
    return pipe(
        prompt=expand_prompt(prompt),
        num_inference_steps=steps or settings.steps,
        guidance_scale=guidance if guidance is not None else settings.guidance,
        width=width,
        height=height,
        generator=torch.Generator("cpu").manual_seed(seed),
    ).images[0]


def read_prompts(path: Path) -> list[str]:
    lines = (line.strip() for line in path.read_text().splitlines())
    return [line for line in lines if line and not line.startswith("#")]


def save_batch(results: dict[str, list[Image.Image]], prompts: list[str], seeds: list[int], out_dir: Path) -> Path:
    """Save images (ordered by prompt, then seed, for each LoRA label) plus a comparison grid."""
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(prompt, seed) for prompt in prompts for seed in seeds]
    for label, images in results.items():
        for (prompt, seed), image in zip(jobs, images):
            image.save(out_dir / f"{slug(label, 32)}__{slug(prompt)}__s{seed}.png")
    if len(results) * len(jobs) > 1:
        rows = [[images[i] for images in results.values()] for i in range(len(jobs))]
        grid(rows, list(results), [f"{prompt} (seed {seed})" for prompt, seed in jobs]).save(
            out_dir / "grid.jpg", quality=90
        )
    print(f"Saved {sum(map(len, results.values()))} images to {out_dir}")
    return out_dir


def slug(text: str, length: int = 48) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:length]


def grid(images: list[list[Image.Image]], col_labels: list[str], row_labels: list[str], tile: int = 384) -> Image.Image:
    """Lay out images[row][col] with column headers and wrapped row captions."""
    header, caption = 36, 72
    sheet = Image.new("RGB", (tile * len(col_labels), header + (tile + caption) * len(row_labels)), "white")
    draw = ImageDraw.Draw(sheet)
    for c, label in enumerate(col_labels):
        draw.text((c * tile + 6, 8), label, fill="black", font_size=20)
    for r, (row, label) in enumerate(zip(images, row_labels)):
        y = header + r * (tile + caption)
        for c, im in enumerate(row):
            sheet.paste(im.resize((tile, tile * im.height // im.width)).crop((0, 0, tile, tile)), (c * tile, y))
        lines = textwrap.wrap(label, tile * len(row) // 9)[:3]
        draw.multiline_text((6, y + tile + 4), "\n".join(lines), fill="black", font_size=17)
    return sheet
