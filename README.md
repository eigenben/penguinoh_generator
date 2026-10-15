# Penguinoh Generator

Generate new images of [Penguinoh the Stuffed Penguin](https://penguinoh.com) with a LoRA fine-tuned on
[FLUX.2 [klein] 4B base](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B) (Apache 2.0).

The LoRA is trained with Hugging Face's DreamBooth script on 52 captioned photos from the penguinoh.com gallery,
and published to [`eigenben/FLUX.2-klein-base-4B-penguinoh`](https://huggingface.co/eigenben/FLUX.2-klein-base-4B-penguinoh).
The default LoRA everywhere is that repo's **`checkpoint-1750`** (`DEFAULT_LORA` in `penguinoh/config.py`), picked
from a comparison of every 250-step checkpoint of the `klein4b-r32-2000` run.
Prompts refer to him as **`pngnh the Stuffed Penguin`**.

## Project layout

```
penguinoh/            Python package and `penguinoh` CLI
  config.py           Trigger phrase, model ids, training defaults, sampler settings
  dataset.py          `penguinoh prepare`: build data/train from website photos
  train.py            `penguinoh train` / `upload`: run the DreamBooth script, push LoRA weights
  generate.py         `penguinoh generate`: render prompts with one or more LoRAs, plus comparison grids
dreambooth/           train_dreambooth_lora_flux2_klein.py, vendored from diffusers v0.41.0
docs/                 Hugging Face model card and sample images
data/train/           Training images + metadata.jsonl (file_name, caption) — the dataset manifest
prompts/eval.txt      Fixed prompts for comparing checkpoints
modal_app.py          Train and generate on Modal GPUs
modal_web.py          Password-protected Gradio UI on Modal
runs/, output/        Local training runs and generated images (git-ignored)
```

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Accounts used: Hugging Face (`HF_TOKEN`, to download models and push the LoRA), Modal (GPUs), and
Weights & Biases (`WANDB_API_KEY`, optional training logs and validation images).

## Dataset

`data/train/metadata.jsonl` lists each training image and its caption. Captions start with
`a photo of pngnh the Stuffed Penguin` and describe the pose, setting and framing — never his appearance,
so that look is learned by the trigger phrase. To add or replace photos, edit the manifest with the website
file name (any extension) and a caption, then rebuild from a checkout of the website repo next to this one:

```bash
uv run penguinoh prepare            # copies, EXIF-rotates, strips metadata, resizes to 1600px
uv run penguinoh prepare --prune    # also delete images no longer in the manifest
```

## Training on Modal

One-time Modal secrets:

```bash
modal secret create huggingface-secret HF_TOKEN=<token>
modal secret create wandb-secret WANDB_API_KEY=<key>
```

Train on an L40S (expect roughly an hour for the default 2,000 steps). `--detach` keeps it running if your laptop sleeps.
Runs are saved to the `penguinoh-generator-runs` volume with a checkpoint every 250 steps.

```bash
uv run modal run --detach modal_app.py::train
uv run modal run --detach modal_app.py::train --max-train-steps 3000 --rank 16 --run-name my-run --push
```

Compare every checkpoint of a run on `prompts/eval.txt`; images and a `grid.jpg` land in `output/<timestamp>/`:

```bash
uv run modal run modal_app.py::generate --run klein4b-20261014-1200 --seeds 1,2
```

Once you've picked a checkpoint, publish the run — the final weights, every checkpoint's LoRA weights (no optimizer
state), `docs/MODEL_CARD.md` as the README, and `docs/samples/` — then point `DEFAULT_LORA` at the chosen checkpoint:

```bash
uv run modal run modal_app.py::upload --run klein4b-r32-2000
```

## Generating

On Modal (no local GPU needed):

```bash
uv run modal run modal_app.py::generate --prompt "surfing a big wave in Hawaii" --seeds 1,2,3
uv run modal run modal_app.py::generate --lora none,eigenben/FLUX.2-klein-base-4B-penguinoh/checkpoint-1750   # with vs without
uv run modal run modal_app.py::generate --sampler distilled                                  # 4-step klein-4B
```

Locally (CUDA, or Apple Silicon via MPS — slow):

```bash
uv run penguinoh generate --prompt "surfing a big wave in Hawaii" --seeds 1 2 3
uv run penguinoh generate --lora eigenben/FLUX.2-klein-base-4B-penguinoh/checkpoint-1500   # a different checkpoint
```

A prompt that doesn't mention `pngnh the Stuffed Penguin` is treated as a scene and becomes
`a photo of pngnh the Stuffed Penguin <scene>`. Name him explicitly for other media, e.g.
`a watercolor painting of pngnh the Stuffed Penguin in Venice`.

Samplers: `base` (FLUX.2 klein base 4B, 50 steps, guidance 4.0, the model the LoRA was trained on) and
`distilled` (FLUX.2 klein 4B, 4 steps, guidance 1.0 — much faster, likeness may be weaker).

## Web UI

```bash
uv run modal deploy modal_app.py
modal secret create penguinoh-web-auth PENGUINOH_WEB_USER=<user> PENGUINOH_WEB_PASSWORD=<password>
uv run modal serve modal_web.py     # temporary URL while running; `modal deploy modal_web.py` to keep it up
```

## Training on another GPU host

```bash
uv sync
uv run --env-file .env penguinoh train --push        # .env holds HF_TOKEN and WANDB_API_KEY
```

Or with Docker (NVIDIA):

```bash
docker build -t penguinoh-generator .
docker run --rm --gpus all --env-file .env -v "$PWD/runs:/app/runs" penguinoh-generator
```

Every training default in `penguinoh/config.py` can be overridden on the command line; see
`uv run penguinoh train --help`.

## Credits

- [FLUX.2 [klein]](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B) by Black Forest Labs
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers) DreamBooth LoRA training
- [Modal](https://modal.com) for GPUs, [Gradio](https://gradio.app) for the UI
