import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from verifiers import accuracy_reward_func, format_reward_func

def evaluate_model(model, tokenizer, test_dataset, num_samples_per_prompt: int = 4, temperature: float = 0.8) -> dict:
    """
    Evaluates model accuracy (Pass@1 and Pass@k) on test prompts.
    """
    model.eval()
    pass1_hits = 0
    passk_hits = 0
    total = len(test_dataset)
    reasoning_lengths = []
    
    for item in test_dataset:
        prompt_text = tokenizer.apply_chat_template(item["prompt"], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        
        # Sample k completions in parallel
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                num_return_sequences=num_samples_per_prompt,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id
            )
            
        generated_texts = []
        for out in outputs:
            gen = tokenizer.decode(out[inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            generated_texts.append([{"content": gen}])
            
            # Extract reasoning length
            match = re.search(r"<think>(.*?)</think>", gen, re.DOTALL)
            if match:
                reasoning_lengths.append(len(match.group(1).split()))
            else:
                reasoning_lengths.append(0)
                
        # Evaluate rewards for the sampled group
        rewards = accuracy_reward_func(generated_texts, ground_truth=[item["ground_truth"]] * num_samples_per_prompt)
        
        # Pass@1 is the outcome of the first sample
        if rewards[0] > 0.5:
            pass1_hits += 1
            
        # Pass@k is hit if ANY sample in the group succeeded
        if any(r > 0.5 for r in rewards):
            passk_hits += 1
            
    return {
        "pass@1": pass1_hits / total,
        f"pass@{num_samples_per_prompt}": passk_hits / total,
        "avg_reasoning_words": sum(reasoning_lengths) / max(len(reasoning_lengths), 1)
    }