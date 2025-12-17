from diffusers.pipelines.auto_pipeline import AutoPipelineForText2Image
import torch
import datetime

pipeline = AutoPipelineForText2Image.from_pretrained(
    "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
).to("cuda")
pipeline.load_lora_weights(
    "eigenben/FLUX.1-dev-penguinoh-generator",
    weight_name="pytorch_lora_weights.safetensors",
)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

image = pipeline("a photo of pngnh the Stuffed Penguin").images[0]
image.save(f"output/inference_{timestamp}_output_base.png")

image = pipeline("a photo of pngnh the Stuffed Penguin in Paris, France with a view of the Eiffel tower").images[0]
image.save(f"output/inference_{timestamp}_output_paris.png")

image = pipeline("a pastel drawing of pngnh the Stuffed Penguin").images[0]
image.save(f"output/inference_{timestamp}_output_pastel.png")

image = pipeline("a goofy drawing of pngnh the Stuffed Penguin").images[0]
image.save(f"output/inference_{timestamp}_output_goofy_drawing.png")

image = pipeline("a photo of pngnh the Stuffed Penguin eating food on the ground").images[0]
image.save(f"output/inference_{timestamp}_output_food.png")

image = pipeline("a photo of pngnh the Stuffed Penguin dining at a nice restaurant").images[0]
image.save(f"output/inference_{timestamp}_output_dining.png")

image = pipeline("a photo of pngnh the Stuffed Penguin in an airplane seat").images[0]
image.save(f"output/inference_{timestamp}_output_airplane.png")

image = pipeline("a photo of pngnh the Stuffed Penguin in an Emirates first class seat drinking champagne").images[0]
image.save(f"output/inference_{timestamp}_output_first_class.png")

image = pipeline("a photo of pngnh the Stuffed Penguin in San Diego, CA at torrey pines").images[0]
image.save(f"output/inference_{timestamp}_output_san_diego.png")

image = pipeline("a photo of pngnh the Stuffed Penguin in San Francisco, CA by the golden gate bridge").images[0]
image.save(f"output/inference_{timestamp}_output_san_francisco.png")
