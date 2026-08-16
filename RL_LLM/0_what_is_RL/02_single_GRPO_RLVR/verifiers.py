import re
import sympy

def format_reward_func(completions: list[list[dict]], **kwargs) -> list[float]:
    """
    Reward 1: Format Integrity Checker
    Checks if the model strictly outputs:
    <think>
    [reasoning]
    </think>
    <answer>
    [final answer]
    </answer>
    """
    # Regex to ensure both tags exist in order and contain non-whitespace text
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    rewards = []
    
    for completion in completions:
        # Extract the assistant's generated string content
        content = completion[0]["content"].strip()
        if re.match(pattern, content, re.DOTALL):
            rewards.append(0.3)  # Format reward bonus
        else:
            rewards.append(0.0)
            
    return rewards

def accuracy_reward_func(completions: list[list[dict]], ground_truth: list[str], **kwargs) -> list[float]:
    """
    Reward 2: Mathematical Accuracy Verifier
    Extracts the string inside <answer>...</answer> and checks numeric/symbolic equality with ground truth.
    """
    rewards = []
    answer_pattern = r"<answer>(.*?)</answer>"
    
    for completion, target in zip(completions, ground_truth):
        content = completion[0]["content"].strip()
        match = re.search(answer_pattern, content, re.DOTALL)
        
        if not match:
            rewards.append(0.0)
            continue
            
        extracted_answer = match.group(1).strip()
        # Clean common formatting like commas or trailing periods
        cleaned_answer = extracted_answer.replace(",", "").rstrip(".")
        
        try:
            # Symbolic check using sympy: handles integer / float equivalence (e.g., 42 == 42.0)
            pred_val = sympy.sympify(cleaned_answer)
            true_val = sympy.sympify(target.strip())
            
            if pred_val == true_val:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        except Exception:
            # Fallback exact string match if sympy cannot parse
            if cleaned_answer == target.strip():
                rewards.append(1.0)
            else:
                rewards.append(0.0)
                
    return rewards

def length_sanity_reward_func(completions: list[list[dict]], **kwargs) -> list[float]:
    """
    Reward 3: Non-empty reasoning check
    Encourages at least some reasoning steps inside <think> tags (>= 20 chars).
    """
    rewards = []
    think_pattern = r"<think>(.*?)</think>"
    
    for completion in completions:
        content = completion[0]["content"].strip()
        match = re.search(think_pattern, content, re.DOTALL)
        if match and len(match.group(1).strip()) >= 20:
            rewards.append(0.1)
        else:
            rewards.append(0.0)
            
    return rewards

if __name__ == "__main__":
    # Quick Unit Test
    test_comp = [[{"content": "<think>Let us multiply 10 by 2, which gives 20.</think>\n<answer>20</answer>"}]]
    test_gt = ["20"]
    
    print("Format Reward:  ", format_reward_func(test_comp))
    print("Accuracy Reward:", accuracy_reward_func(test_comp, ground_truth=test_gt))
    print("Length Reward:  ", length_sanity_reward_func(test_comp))