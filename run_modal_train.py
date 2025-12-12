import modal
from main import TrainConfig, download, train

MODEL_DIR = "/workspace/model"
IMAGES_DIR = "/workspace/data/images"

app = modal.App(name="penguinoh-generator")
image = modal.Image.debian_slim(python_version="3.12").apt_install("git").uv_sync()
model_volume = modal.Volume.from_name("penguinoh-generator-model-volume", create_if_missing=True)
images_volume = modal.Volume.from_name("penguinoh-generator-images-volume", create_if_missing=True)

image = image.env(
    {"HF_XET_HIGH_PERFORMANCE": "1"}  # turn on faster downloads from HF
)

image = image.add_local_file("./main.py", "/root/main.py")
image = image.add_local_dir("./dreambooth", "/root/dreambooth")

huggingface_secret = modal.Secret.from_name(
    "huggingface-secret", required_keys=["HF_TOKEN"]
)

wandb_secret = modal.Secret.from_name(
    "wandb-secret", required_keys=["WANDB_API_KEY"]
)

@app.function(
    volumes={MODEL_DIR: model_volume, IMAGES_DIR: images_volume},
    image=image,
    secrets=[huggingface_secret, wandb_secret],
    timeout=600,  # 10 minutes
)
def run_download(model_dir, config):
    download(model_dir, config)

@app.function(
    volumes={MODEL_DIR: model_volume, IMAGES_DIR: images_volume},
    gpu="A100-80GB",
    image=image,
    secrets=[huggingface_secret, wandb_secret],
    timeout=600,  # 10 minutes
)
def run_train(model_dir, images_dir, config):
    train(model_dir, images_dir, config)
    model_volume.commit()

@app.local_entrypoint()
def run(
    max_train_steps: int = 250,
):
    print("🎨 loading model")
    run_download.remote(MODEL_DIR, TrainConfig())
    
    print("🎨 setting up training")
    config = TrainConfig(max_train_steps=max_train_steps)
    
    run_train.remote(MODEL_DIR, IMAGES_DIR, config)
    print("🎨 training finished")
