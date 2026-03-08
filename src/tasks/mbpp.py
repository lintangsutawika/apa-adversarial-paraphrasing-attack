import re

from datasets import load_dataset

from rllm.data.dataset import DatasetRegistry
from rllm.rewards.reward_types import RewardOutput

from src.tasks.code_eval import check_correctness, compute_code_eval

stop_words=["\nclass", "\nassert", '\n"""', "\nprint", "\nif", "\n<|/", "\n```"]
start_words=["```python\n", "```\n"]

ATTACK_SYS_PROMPT = (
    "Rewrite the given programming problem. "
    "Do not write any code, your task is to only output a rewritten version of "
    "the problem description inside <problem></problem>."
)

def prepare_mbpp_data():
    mbpp_dataset = load_dataset("google-research-datasets/mbpp")
    train_dataset = mbpp_dataset["train"]
    test_dataset = mbpp_dataset["test"]

    def preprocess_fn(example, idx):
        text = example["text"].strip()
        all_tests = example["test_list"] + example["challenge_test_list"]
        test_cases = '\n'.join(all_tests)
        if example["test_setup_code"]:
            test_cases = example["test_setup_code"] + "\n\n" + test_cases
        data = {
            "data_source": "mbpp",
            "question": ATTACK_SYS_PROMPT + "\n\nProblem: " + text,
            "target_prompts": [
                {"role": "system", "content": "Solve the following programming problem. Write code that can pass the provided test cases."},
                {"role": "user", "content": "__PARAPHRASED_QUESTION__" + "\n\n" + test_cases},
            ],
            "ability": "coding",
            "reward_model": {"style": "rule", "ground_truth": None},
            "extra_info": {
                "index": idx,
                "task_id": example["task_id"],
                "original_problem": text,
                "solution": example["code"],
                "test_cases": all_tests
            },
        }
        return data

    train_dataset = train_dataset.map(preprocess_fn, with_indices=True)
    test_dataset = test_dataset.map(preprocess_fn, with_indices=True)

    train_dataset = DatasetRegistry.register_dataset("mbpp", train_dataset, "train")
    test_dataset = DatasetRegistry.register_dataset("mbpp", test_dataset, "test")
    return train_dataset, test_dataset

def stop_at_stop_token(decoded_string, stop_tokens):
    """
    Produces the prefix of decoded_string that ends at the first occurrence of
    a stop_token.
    WARNING: the decoded_string *must not* include the prompt, which may have stop tokens
    itself.
    """
    min_stop_index = len(decoded_string)
    for stop_token in stop_tokens:
        stop_index = decoded_string.find(stop_token)
        if stop_index != -1 and stop_index < min_stop_index:
            min_stop_index = stop_index
    return decoded_string[:min_stop_index]

def mbpp_reward_fn(task_info: dict, response: str) -> float:

    if response is None:
        return RewardOutput(reward=0.0, metadata={"error": "No response generated"})

    if isinstance(response, str):
        response = [response]

    code_candidates = []
    for resp in response:
        for start_token in start_words:
            if start_token in resp:
                resp = resp.split(start_token)[-1]
                break
        # Clean up the response to extract code
        code = stop_at_stop_token(resp, stop_words).strip()
        code_candidates.append(code)

    score, pass_at_k, results = compute_code_eval(
        references=[task_info["extra_info"]["test_cases"]],
        predictions=[code_candidates],
        k=[1]
    )

    return RewardOutput(reward=score, metadata={
        "pass_at_k": pass_at_k,
        "results": results,
    })


if __name__ == "__main__":
    train_dataset, test_dataset = prepare_mbpp_data()
    print(train_dataset)
    print(test_dataset)
