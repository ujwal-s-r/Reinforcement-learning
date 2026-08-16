from datasets import Dataset

# System prompt instructing the model on the required reasoning and answer contract
SYSTEM_PROMPT = """You are a helpful math reasoning assistant.
You must think step-by-step before answering.
Put your intermediate chain-of-thought inside <think>...</think> tags.
Put your final numerical answer strictly inside <answer>...</answer> tags.
Example:
<think>
To solve 5 * 6, we multiply 5 by 6 which equals 30.
</think>
<answer>30</answer>
"""

def generate_math_dataset(num_samples: int = 120, split_ratio: float = 0.8) -> tuple[Dataset, Dataset]:
    """
    Generates a curated dataset of multi-step arithmetic problems.
    Each sample contains:
      - 'prompt': Formatted conversation prompt with system prompt & question
      - 'ground_truth': Clean numerical string for verification
    """
    import random
    random.seed(42)
    
    samples = []
    for _ in range(num_samples):
        a = random.randint(10, 99)
        b = random.randint(5, 50)
        c = random.randint(2, 20)
        
        op_type = random.choice(["add_sub", "mult_add", "mult_sub"])
        if op_type == "add_sub":
            ans = a + b - c
            question = f"What is ({a} + {b}) - {c}?"
        elif op_type == "mult_add":
            ans = (a * 2) + b
            question = f"What is ({a} * 2) + {b}?"
        else:
            ans = (b * c) - a
            question = f"What is ({b} * {c}) - {a}?"
            
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ]
        
        samples.append({
            "prompt": prompt,
            "ground_truth": str(ans),
            "question": question
        })
    
    split_idx = int(num_samples * split_ratio)
    train_data = samples[:split_idx]
    test_data = samples[split_idx:]
    
    return Dataset.from_list(train_data), Dataset.from_list(test_data)

if __name__ == "__main__":
    train_ds, test_ds = generate_math_dataset(10)
    print(f"Generated {len(train_ds)} train and {len(test_ds)} test samples.")
    print("Sample Prompt:\n", train_ds[0]["prompt"])
    print("Sample Ground Truth:", train_ds[0]["ground_truth"])