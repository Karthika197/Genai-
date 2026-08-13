# Ex.No 2 - Prompt Engineering: Zero-shot, Few-shot, Chain-of-Thought
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

# 1. Zero-shot prompt
zero_shot_prompt = (
    "Classify the sentiment of this review as Positive or Negative: "
    "'The product quality is excellent!'\nSentiment:"
)

# 2. Few-shot prompt
few_shot_prompt = """Review: 'I loved this movie, it was fantastic.'
Sentiment: Positive
Review: 'The service was slow and disappointing.'
Sentiment: Negative
Review: 'The product quality is excellent!'
Sentiment:"""

# 3. Chain-of-Thought prompt
cot_prompt = """Q: A shop had 15 apples. It sold 6 and then received 10 more. How many apples now?
A: Let's think step by step. 15 - 6 = 9. 9 + 10 = 19. The answer is 19.

Q: A library had 120 books. It lent out 45 and bought 30 new books. How many books now?
A: Let's think step by step."""

prompts = {
    "Zero-shot": zero_shot_prompt,
    "Few-shot": few_shot_prompt,
    "Chain-of-Thought": cot_prompt
}

for name, p in prompts.items():
    out = generator(
        p,
        max_new_tokens=25,
        num_return_sequences=1,
        do_sample=False,
        truncation=True,
        pad_token_id=generator.tokenizer.eos_token_id
    )
    print(f"=== {name} ===")
    print(out[0]["generated_text"])
    print()
