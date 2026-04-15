import re

from datasets import load_dataset

from rllm.data.dataset import DatasetRegistry


# Adapted from verl/examples/data_preprocess/gsm8k.py
def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


# Stronger wording for the current diagnostic run: preserve the mathematical
# task exactly, but push the model toward subtler target-specific rewrites.
ATTACK_SYS_PROMPT = """# Objective
Create a new version of the original math problem that preserves the required mathematical reasoning, solution path, and final answer, while increasing the likelihood that the specific target model fails even though humans and reference models can still solve it correctly with the same single valid answer.

# Instructions
- Rewrite the original problem so it requires the same mathematical reasoning and leads to the same solution and answer.
- Do not include instructions about response formatting in the rewritten problem, since models will receive separate system instructions.
- Use insights from successful attack examples to identify reproducible mistake patterns when available.
- Exploit theorized weaknesses of the target model subtly.
- The rewritten problem does not need to be more challenging, more complex, or more vague.
- Ensure the rewritten problem still has a single valid answer and remains correctly solvable by humans and reference models.

# Output Requirements
Your response MUST contain only the rewritten math problem, enclosed exactly as follows:

```text
<problem>...</problem>
```
"""


def prepare_gsm8k_data():
    gsm8k_dataset = load_dataset("openai/gsm8k", "main")
    train_dataset = gsm8k_dataset["train"]
    test_dataset = gsm8k_dataset["test"]

    def preprocess_fn(example, idx):
        return {
            "messages": [
                {"role": "system", "content": ATTACK_SYS_PROMPT},
                {"role": "user", "content": example["question"]},
            ],
            "ground_truth": extract_solution(example["answer"]),
            "data_source": "gsm8k",
            "target_prompts": [
                {
                    "role": "system",
                    "content": "Reason step by step and put your final answer within \\boxed{}.",
                },
                {"role": "user", "content": "__PARAPHRASED_QUESTION__"},
            ],
        }

    # Force remap so prompt changes in this file are reflected immediately during debugging.
    train_dataset = train_dataset.map(
        preprocess_fn, with_indices=True, load_from_cache_file=False
    )
    test_dataset = test_dataset.map(
        preprocess_fn, with_indices=True, load_from_cache_file=False
    )

    # Materialize to plain Python records before registration so the registered
    # parquet reflects the freshly mapped prompt content.
    train_dataset = train_dataset.to_list()
    test_dataset = test_dataset.to_list()

    train_dataset = DatasetRegistry.register_dataset("gsm8k", train_dataset, "train")
    test_dataset = DatasetRegistry.register_dataset("gsm8k", test_dataset, "test")
    return train_dataset, test_dataset


if __name__ == "__main__":
    train_dataset, test_dataset = prepare_gsm8k_data()
    print(train_dataset)
    print(test_dataset)
