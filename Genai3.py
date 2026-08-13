# Ex.No 3 - Conversational AI Chatbot using DialoGPT
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

chat_history_ids = None

# Simulated multi-turn conversation (predefined inputs for notebook reproducibility)
user_inputs = ["Hi, how are you?", "What can you help me with?", "quit"]

print("Chatbot ready! Type 'quit' to exit.\n")

for step, user_input in enumerate(user_inputs):
    if user_input.lower() == "quit":
        break

    print(f">> User: {user_input}")

    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    bot_input_ids = (
        torch.cat([chat_history_ids, new_input_ids], dim=-1)
        if chat_history_ids is not None else new_input_ids
    )

    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.9
    )

    response = tokenizer.decode(
        chat_history_ids[:, bot_input_ids.shape[-1]:][0],
        skip_special_tokens=True
    )
    print(f"Bot: {response}\n")
