# Ex.No 8 - Image Generation using Stable Diffusion
from diffusers import StableDiffusionPipeline
import torch

# Load pipeline (fp16 requires GPU runtime; fallback to fp32 on CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=dtype
)
pipe = pipe.to(device)

prompt = "A futuristic city skyline at sunset, digital art, highly detailed"

image = pipe(
    prompt,
    num_inference_steps=30,
    guidance_scale=7.5
).images[0]

image.save("generated_city.png")
print("Image generated and saved as generated_city.png")
