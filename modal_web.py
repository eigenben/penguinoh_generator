"""Password-protected Gradio UI that calls the deployed Generator from modal_app.py.

    modal deploy modal_app.py      # once, and after changing the generator
    modal secret create penguinoh-web-auth PENGUINOH_WEB_USER=<user> PENGUINOH_WEB_PASSWORD=<password>
    modal serve modal_web.py       # temporary URL while running; `modal deploy modal_web.py` to keep it up
"""

import io

import modal

app = modal.App(name="penguinoh-generator-web")
image = modal.Image.debian_slim(python_version="3.12").uv_sync().add_local_python_source("penguinoh")


@app.function(image=image, secrets=[modal.Secret.from_name("penguinoh-web-auth")], max_containers=1)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def web():
    import os

    import gradio as gr
    from fastapi import FastAPI
    from PIL import Image

    from penguinoh.config import DEFAULT_LORA, TRIGGER

    generator = modal.Cls.from_name("penguinoh-generator", "Generator")()

    def go(scene: str, seed: float):
        scene = scene.strip() or "waddling through a snowy forest at dusk"
        png = generator.generate.remote(DEFAULT_LORA, [scene], [int(seed)])[0]
        return Image.open(io.BytesIO(png))

    with gr.Blocks(title="Penguinoh generator") as ui:
        gr.Markdown(
            f"# Generate images of {TRIGGER}\n\n"
            "Describe a scene (\"surfing in Hawaii\"), or write a full prompt that names "
            f"**{TRIGGER}** for other styles (\"a watercolor painting of {TRIGGER} in Venice\")."
        )
        with gr.Row():
            with gr.Column():
                scene = gr.Textbox(label="Scene or prompt", lines=4)
                seed = gr.Number(label="Seed", value=1, precision=0)
                button = gr.Button("Generate", variant="primary")
            output = gr.Image(label="Result", height=640)
        button.click(go, inputs=[scene, seed], outputs=output)

    auth = (os.environ["PENGUINOH_WEB_USER"], os.environ["PENGUINOH_WEB_PASSWORD"])
    return gr.mount_gradio_app(FastAPI(), ui, path="/", auth=auth)
