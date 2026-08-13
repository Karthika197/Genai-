# Ex.No 7 - AI-Powered Code Generation and Debugging using CodeGen
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-350M-mono")

def generate_code(prompt, max_new_tokens=80):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    output = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# 1. Code generation from a natural-language instruction
prompt1 = "# Write a Python function to check if a number is prime\ndef is_prime(n):"
print("Generated Function:\n", generate_code(prompt1))

# 2. Debugging a faulty snippet
buggy_code = """# The following function should return the factorial of n, but has a bug. Fix it.
def factorial(n):
    result = 0
    for i in range(1, n+1):
        result = result * i
    return result

# Corrected function:
def factorial_fixed(n):"""

print("\nDebug Suggestion:\n", generate_code(buggy_code, max_new_tokens=60))
