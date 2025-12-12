from dataclasses import dataclass
import modal
from main import SharedConfig, TrainConfig, download, train

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


@app.cls(image=image, gpu="A100", volumes={MODEL_DIR: model_volume})
class Model:
    @modal.enter()
    def load_model(self):
        import torch
        from diffusers import DiffusionPipeline

        # Reload the modal.Volume to ensure the latest state is accessible.
        model_volume.reload()

        # set up a hugging face inference pipeline using our model
        pipe = DiffusionPipeline.from_pretrained(
            MODEL_DIR,
            torch_dtype=torch.bfloat16,
        ).to("cuda")
        pipe.load_lora_weights(MODEL_DIR)
        self.pipe = pipe

    @modal.method()
    def inference(self, text, config):
        image = self.pipe(
            text,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
        ).images[0]

        return image

@dataclass
class AppConfig(SharedConfig):
    num_inference_steps: int = 50
    guidance_scale: float = 6


@app.function(
    image=image,
    max_containers=1,
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def fastapi_app():
    import gradio as gr
    from fastapi import FastAPI
    from fastapi.responses import FileResponse
    from gradio.routes import mount_gradio_app

    web_app = FastAPI()

    # Call out to the inference in a separate Modal environment with a GPU
    def go(text=""):
        if not text:
            text = example_prompts[0]
        return Model().inference.remote(text, config)

    config = AppConfig()
    instance_phrase = f"{config.instance_name} the {config.class_name}"
    description = f"Describe what they are doing or how a particular artist or style would depict them. Be fantastical!"
    theme = gr.themes.Default(primary_hue="green", secondary_hue="emerald", neutral_hue="neutral")

    with gr.Blocks(theme=theme, title=f"Generate images of Penguinoh") as interface:
        gr.Markdown(f"# Generate images of {instance_phrase}.\n\n{description}")
        with gr.Row():
            inp = gr.Textbox(
                label="",
                placeholder=f"Describe the version of {instance_phrase} you'd like to see",
                lines=10,
            )
            out = gr.Image(height=512, width=512, label="", min_width=512, elem_id="output")
        with gr.Row():
            btn = gr.Button("Generate", variant="primary", scale=2)
            btn.click(fn=go, inputs=inp, outputs=out)

    return mount_gradio_app(
        app=web_app,
        blocks=interface,
        path="/",
    )

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
