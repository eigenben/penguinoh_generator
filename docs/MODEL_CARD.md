---
base_model: black-forest-labs/FLUX.2-klein-base-4B
library_name: diffusers
license: cc-by-nc-4.0
instance_prompt: a photo of pngnh the Stuffed Penguin
tags:
- text-to-image
- diffusers
- lora
- flux2-klein
- dreambooth
widget:
- text: a studio portrait photo of pngnh the Stuffed Penguin, soft lighting, plain gray background
  output:
    url: samples/studio-portrait.jpg
- text: a photo of pngnh the Stuffed Penguin in front of the Golden Gate Bridge on a foggy morning, upper body
  output:
    url: samples/golden-gate.jpg
- text: a photo of pngnh the Stuffed Penguin seen from the side, walking across Shibuya crossing in Tokyo at night with neon signs
  output:
    url: samples/shibuya.jpg
- text: a watercolor painting of pngnh the Stuffed Penguin riding in a gondola in Venice
  output:
    url: samples/watercolor-venice.jpg
- text: a claymation still of pngnh the Stuffed Penguin baking bread in a cozy kitchen
  output:
    url: samples/claymation-kitchen.jpg
- text: a vintage 1960s travel poster of pngnh the Stuffed Penguin in front of the Eiffel Tower, bold flat colors, the text "PARIS"
  output:
    url: samples/paris-poster.jpg
---

# Penguinoh — FLUX.2 [klein] 4B LoRA

A DreamBooth LoRA for [FLUX.2 [klein] base 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B)
that generates [Penguinoh](https://penguinoh.com), a well-traveled stuffed penguin.

<Gallery />

## Trigger phrase

Refer to him as **`pngnh the Stuffed Penguin`**, e.g. `a photo of pngnh the Stuffed Penguin surfing in Hawaii`
or `a watercolor painting of pngnh the Stuffed Penguin in Venice`.

## Weights

The repo-root `pytorch_lora_weights.safetensors` is **step 1750**, which had the best likeness while keeping varied
compositions in a comparison of every checkpoint. Each 250-step checkpoint, including the final step 2000, is also
in its own `checkpoint-N/` folder; likeness is unreliable before step 1250.

You can try it from this page's inference widget (billed to your Hugging Face account through an inference
provider), or run it yourself with diffusers:

```python
import torch
from diffusers import Flux2KleinPipeline

pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-base-4B", torch_dtype=torch.bfloat16).to("cuda")
pipe.load_lora_weights("eigenben/FLUX.2-klein-base-4B-penguinoh", weight_name="pytorch_lora_weights.safetensors")
image = pipe(
    "a photo of pngnh the Stuffed Penguin sitting on a park bench in Central Park in autumn",
    num_inference_steps=50,
    guidance_scale=4.0,
    generator=torch.Generator("cpu").manual_seed(1),
).images[0]
```

## Training

- **Data:** 52 photos from the penguinoh.com gallery, chosen for variety in pose, distance, angle and setting,
  each with its own caption describing the scene and framing (never his appearance).
- **Method:** diffusers `train_dreambooth_lora_flux2_klein.py` (v0.41.0), 1024px aspect-ratio buckets,
  LoRA rank 32 / alpha 32, AdamW at 1e-4 with 100 warmup steps, batch size 1, 2000 steps, bf16, on one L40S.
- **Code:** [github.com/eigenben/penguinoh_generator](https://github.com/eigenben/penguinoh_generator)

## Limitations

- Illustration styles such as children's-book art drift toward a generic cartoon penguin.
- The trigger word can leak into rendered text (e.g. "PNGNH" on a poster), mostly in earlier checkpoints.
- Later checkpoints repeat compositions more for the same prompt; vary seeds for diversity.

## License

The LoRA weights and sample images are © Ben Hughes, licensed under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). The base model is licensed separately by
Black Forest Labs (Apache 2.0); the training code is Apache 2.0.
