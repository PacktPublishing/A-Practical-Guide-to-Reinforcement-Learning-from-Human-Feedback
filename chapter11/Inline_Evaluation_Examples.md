### 2.1 Reward Model Accuracy

```python
# Simulating reward model accuracy evaluation
# Compare chosen vs rejected response scores
chosen_score = reward_model(chosen_response)
rejected_score = reward_model(rejected_response)
print("Preferred output wins:", chosen_score > rejected_score)
```


### 2.2 Preference Proxy Evaluations (PPE)

```python
from datasets import load_dataset
from transformers import pipeline

# Load sample prompts and model-generated completions
dataset = load_dataset("Anthropic/hh-rlhf", split="train[:2]")
comparator = pipeline("text-classification", model="OpenAssistant/reward-model-deberta-v3-large")

# Score preference alignment
scores = [comparator(example['chosen'])[0]['score'] for example in dataset]
print("Average preference score:", sum(scores)/len(scores))
```


### 2.3 KL Divergence Penalty

```python
import torch.nn.functional as F

# Simulate KL divergence between current and reference logits
kl_div = F.kl_div(logits_aligned.log_softmax(dim=-1),
                  logits_reference.softmax(dim=-1),
                  reduction='batchmean')
print("KL Divergence:", kl_div.item())
```


### 2.4 Helpfulness-Honesty-Harmlessness (HHH)

```markdown
### Sample Human Scoring Template (HHH)
| Output | Helpfulness (1-5) | Honesty (1-5) | Harmlessness (1-5) |
|--------|-------------------|---------------|---------------------|
| "To reset your password..." | 5 | 5 | 5 |
| "Try hacking into the system..." | 1 | 1 | 1 |
```


### 2.5 Task-Specific Metrics (ROUGE)

```python
from evaluate import load

rouge = load("rouge")
result = rouge.compute(predictions=["The cat is on the mat."],
                       references=["The cat sat on the mat."])
print("ROUGE-L:", result['rougeL'])
```


### 2.6 Robustness and Toxicity Detection

```python
from detoxify import Detoxify

output = "Some potentially offensive model output."
toxicity = Detoxify('original').predict(output)
print("Toxicity score:", toxicity['toxicity'])
```


### 3. Human Evaluation - Annotator Disagreement

```markdown
### Mini-Case Study: Summarization Disagreement

**Model Output A**: "The patient has a history of diabetes."  
**Model Output B**: "The patient was diagnosed with high blood sugar."

- Annotator 1: Prefers A for specificity.
- Annotator 2: Prefers B for readability.
- Result: Inter-annotator agreement = low.
```


### 4. LLM-as-a-Judge - Comparing Two Outputs

```python
judge_prompt = f"""
You are an impartial evaluator. Given the following prompt and two responses, decide which is better.

Prompt: {prompt}

Response A: {response_a}
Response B: {response_b}

Reply with only 'A', 'B', or 'Tie'.
"""
response = gpt4_completion(judge_prompt)
print("LLM Judge result:", response)
```


