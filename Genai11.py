# Ex.No 11 - Integrated Multimedia Content Generation Pipeline
from transformers import pipeline
from diffusers import StableDiffusionPipeline
from gtts import gTTS
import torch

topic = "The benefits of renewable energy"

# 1. Text generation
text_generator = pipeline("text2text-generation", model="google/flan-t5-base")
text_prompt = f"Write a short, engaging paragraph about: {topic}"
generated_text = text_generator(text_prompt, max_length=80)[0]["generated_text"]
print("Generated Text:\n", generated_text)

# 2. Image generation (derived prompt)
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

image_prompt = f"An illustration representing {topic}, digital art"
sd_pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=dtype
).to(device)

image = sd_pipe(image_prompt, num_inference_steps=25).images[0]
image.save("content_image.png")
print("Image saved as content_image.png")

# 3. Audio generation (text-to-speech)
tts = gTTS(text=generated_text, lang="en")
tts.save("content_audio.mp3")
print("Audio saved as content_audio.mp3")
