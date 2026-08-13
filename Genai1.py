# Ex.No 1 - Text Generation Using Pre-Trained Foundation Models (GPT-2)
from transformers import pipeline, set_seed

# Load pre-trained GPT-2 text-generation pipeline
generator = pipeline("text-generation", model="gpt2")
set_seed(42)

# Input prompt
prompt = "Artificial Intelligence will transform the future of"

# Generate text using sampling-based decoding (top-k + top-p)
outputs = generator(
    prompt,
    max_length=60,
    num_return_sequences=2,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    do_sample=True,
    truncation=True
)

for i, out in enumerate(outputs, 1):
    print(f"--- Generated Text {i} ---")
    print(out["generated_text"])
    print()

# Compare with greedy decoding (deterministic)
greedy_output = generator(prompt, max_length=60, do_sample=False, truncation=True)
print("--- Greedy Decoding Output ---")
print(greedy_output[0]["generated_text"])
