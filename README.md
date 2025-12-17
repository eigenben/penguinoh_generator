# Penguinoh Generator

A personalized AI image generator for creating custom images of [Penguinoh the Stuffed Penguin](http://penguinoh.com) using DreamBooth fine-tuning and the FLUX.1-dev diffusion model.

## Overview

This project uses DreamBooth to fine-tune the FLUX.1-dev text-to-image model on custom images of a stuffed penguin, enabling the generation of new, creative images featuring the character in various scenarios, styles, and settings.

The project supports both local training and cloud deployment via [Modal](https://modal.com), providing a scalable infrastructure for GPU-accelerated training and inference.

## Features

- **DreamBooth Fine-tuning**: Train a LoRA adapter on custom images using Hugging Face's DreamBooth implementation
- **FLUX.1-dev Model**: Uses Black Forest Labs' state-of-the-art diffusion model as the base
- **Local Inference**: Generate images locally using pre-trained models from Hugging Face Hub
- **Cloud Deployment**: Deploy training and inference on Modal with A100 GPUs
- **Web Interface**: Interactive Gradio-based UI for generating images
- **Training Monitoring**: Integration with Weights & Biases for experiment tracking
- **FastAPI Backend**: REST API endpoints for programmatic access

## Project Structure

```
.
├── main.py              # Local training script (download & train)
├── inference.py         # Local inference script using pre-trained model
├── modal_app.py         # Modal deployment (cloud training & inference)
├── pyproject.toml       # Project dependencies
├── data/
│   └── images/          # Training images of the subject
├── dreambooth/          # Hugging Face DreamBooth training scripts
├── model/               # Output directory for trained models
└── output/              # Generated images from inference
```

## Requirements

- Python 3.12+
- CUDA-compatible GPU (recommended: A100) for training
- Hugging Face account and API token
- Modal account (for cloud deployment)
- Weights & Biases account (optional, for training monitoring)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd penguinoh_generator
```

2. Install dependencies using uv (or pip):
```bash
uv sync
```

3. Set up your API keys:
   - Create a `.env` file or export environment variables:
   ```bash
   export HF_TOKEN=<your-huggingface-token>
   export WANDB_API_KEY=<your-wandb-key>  # Optional
   ```

## Usage

### Local Training

0. **Ensure uv dependencies are installed:**
```
uv sync
```

1. **Download the base model**:
```bash
uv run main.py download
```

2. **Train the model**:
```bash
uv run main.py train
```

3. **Download and train in one step**:
```bash
uv run main.py
```

### Local Inference

Run inference using the pre-trained model from Hugging Face Hub:

1. **Ensure the output directory exists**:
```bash
mkdir -p output
```

2. **Run the inference script**:
```bash
uv run inference.py
```

This will:
- Load the FLUX.1-dev base model with bfloat16 precision
- Load the fine-tuned LoRA weights from `eigenben/FLUX.1-dev-penguinoh-generator`
- Generate 10 example images with various prompts (Paris, San Francisco, pastel drawing, etc.)
- Save timestamped images to the `output/` directory

You can modify `inference.py` to customize the prompts and generate your own images.

### Cloud Training with Modal

1. **Set up Modal secrets**:
```bash
modal secret create huggingface-secret HF_TOKEN=<your-token>
modal secret create wandb-secret WANDB_API_KEY=<your-key>  # Optional
```

2. **Upload training images to Modal volume**:
```bash
# Upload your images to the Modal volume at /workspace/data/images
```

3. **Run training on Modal**:
```bash
uv run modal run modal_app.py --max-train-steps 250
```

4. **Deploy the web interface**:
```bash
uv run modal deploy modal_app.py
```

This will create a public URL for the Gradio interface where you can generate images.

## Run on a remote GPU host
- Ensure rsync is installed: `apt update && apt install rsync -y`
- Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Sync project to remote host: `bin/rspec 1.1.1.1:22 /workspace` (`/workspace` is implied/default)
- Ensure HF_TOKEN and other env vars are set in `.env` on remote host
- SSH into remote host and run: `cd /workspace && uv sync && uv run --env-file .env python main.py`

## Docker (GPU)
- Build: `docker build -t penguinoh-generator:latest .`
- Ensure HF_TOKEN and other env vars are set in `.env` on remote host
- Run (NVIDIA): `docker run --rm --gpus all --env-file .env -v "$PWD/model:/app/model" penguinoh-generator:latest`

## Configuration

Training parameters can be configured in `main.py`:

```python
@dataclass
class TrainConfig(SharedConfig):
    instance_name: str = "pngnh"           # Instance identifier
    class_name: str = "Stuffed Penguin"    # Class description
    model_name: str = "black-forest-labs/FLUX.1-dev"
    resolution: int = 512                   # Training resolution
    train_batch_size: int = 3              # Batch size
    rank: int = 16                         # LoRA rank
    learning_rate: float = 4e-4
    max_train_steps: int = 500             # Training iterations
    seed: int = 117                        # Random seed
```

Inference parameters are in `modal_app.py`:

```python
@dataclass
class AppConfig(SharedConfig):
    num_inference_steps: int = 50          # Quality vs speed tradeoff
    guidance_scale: float = 6              # Prompt adherence strength
```

## How It Works

1. **Download**: Fetches the FLUX.1-dev base model from Hugging Face
2. **Training**: Uses DreamBooth with LoRA to fine-tune the model on custom images
   - Creates a unique instance phrase: "pngnh the Stuffed Penguin"
   - Trains the model to recognize and generate this specific character
   - Saves LoRA weights for efficient storage and loading
3. **Inference**: Loads the base model + LoRA weights to generate new images
4. **Web Interface**: Provides an easy-to-use Gradio UI for image generation

## Training Details

- **Method**: DreamBooth with LoRA (Low-Rank Adaptation)
- **Base Model**: FLUX.1-dev (from Black Forest Labs)
- **Precision**: BFloat16 (when CUDA available)
- **Optimizer**: AdamW with configurable learning rate
- **Scheduler**: Constant learning rate (configurable)
- **GPU**: A100-80GB recommended for training

## Web Interface

The deployed Gradio interface allows users to:
- Enter text prompts describing the desired image
- Generate 512x512 images featuring Penguinoh
- Experiment with different artistic styles and scenarios

Example prompts:
- "pngnh the Stuffed Penguin in space"
- "pngnh the Stuffed Penguin painted by Van Gogh"
- "pngnh the Stuffed Penguin as a superhero"

## Credits

- Built with [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- Deployed on [Modal](https://modal.com)
- UI powered by [Gradio](https://gradio.app)
- Base model: [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)
