import os
import re
from threading import BoundedSemaphore
from typing import Callable

from rllm.rewards.reward_types import RewardOutput

import litellm

kwargs = {
    "base_url": os.environ.get("LLM_API_URL", "http://localhost:8000"),
    "api_key": os.environ.get("LLM_API_KEY", "fake_api_key_for_testing"),
}

_LLM_CONCURRENCY = int(os.environ.get("LLM_MAX_CONCURRENCY", "1"))
_LLM_CALL_SEMAPHORE = BoundedSemaphore(value=_LLM_CONCURRENCY)


def _extract_paraphrased_question(action: str) -> str | None:
    cleaned = action.replace("<|im_end|>", "").strip()
    cleaned = re.sub(r"^<think>\s*</think>\s*", "", cleaned, flags=re.DOTALL).strip()

    match = re.search(r"<problem>(.*?)</problem>", cleaned, re.DOTALL)
    if match:
        return match.group(1).strip()

    return cleaned or None


def _completion_content(model: str, messages: list[dict]) -> str | None:
    # Fail loudly here so provider/config errors do not masquerade as valid zero-reward samples.
    with _LLM_CALL_SEMAPHORE:
        return (
            litellm.completion(model=model, messages=messages, **kwargs)
            .choices[0]
            .message.content.strip()
        )


def adversarial_reward_fn(
    task_reward_fn: Callable[[dict, str | None], RewardOutput],
    task_info: dict,
    action: str,
) -> RewardOutput:
    """
    A reward function for math tasks that implements the RewardFunction protocol.

    Args:
        task_reward_fn: The reward function specific to the task
        task_info: A dictionary containing information about the task
        action: The agent's response/solution

    Returns:
        float: The calculated reward value based on math evaluation
    """

    paraphrased_question = _extract_paraphrased_question(action)

    if paraphrased_question is None:
        return RewardOutput(reward=0.0, metadata={"error": "Invalid action format"})

    messages = [dict(message) for message in task_info["target_prompts"]]
    messages[-1]["content"] = messages[-1]["content"].replace(
        "__PARAPHRASED_QUESTION__", paraphrased_question
    )

    # Current branch/runtime expects provider selection through env or task_info, not hardcoded defaults.
    victim_model = task_info.get("victim_model") or os.environ.get("LLM_VICTIM_MODEL")
    assert victim_model, (
        "Missing victim model. Set task_info['victim_model'] or LLM_VICTIM_MODEL."
    )

    reference_models = task_info.get(
        "reference_models", task_info.get("reference_model")
    )
    if reference_models is None:
        reference_models = os.environ.get("LLM_REFERENCE_MODELS") or os.environ.get(
            "LLM_REFERENCE_MODEL"
        )
    if isinstance(reference_models, str):
        reference_models = [
            model.strip() for model in reference_models.split(",") if model.strip()
        ]
    assert reference_models, (
        "Missing reference model(s). Set task_info['reference_models'] or "
        "LLM_REFERENCE_MODEL(S)."
    )

    victim_response = _completion_content(victim_model, messages)
    reference_responses = [
        _completion_content(reference_model, messages)
        for reference_model in reference_models
    ]

    victim_reward = task_reward_fn(task_info, victim_response)
    reference_rewards = [
        task_reward_fn(task_info, reference_response)
        for reference_response in reference_responses
    ]
    # Completion failures are transport errors, not successful adversarial wins.
    completion_failed = victim_response is None or any(
        reference_response is None for reference_response in reference_responses
    )

    if (
        not completion_failed
        and victim_reward.reward < 1
        and all(reference_reward.reward == 1 for reference_reward in reference_rewards)
    ):
        reward = 1.0
    else:
        reward = 0.0

    return RewardOutput(
        reward=reward,
        metadata={
            "paraphrased_question": paraphrased_question,
            "victim_model": victim_model,
            "reference_models": reference_models,
            "victim_response": victim_response,
            "reference_responses": reference_responses,
            "victim_reward": victim_reward.reward,
            "victim_reward_metadata": victim_reward.metadata,
            "completion_failed": completion_failed,
            "reference_rewards": [
                reference_reward.reward for reference_reward in reference_rewards
            ],
            "reference_reward_metadata": [
                reference_reward.metadata for reference_reward in reference_rewards
            ],
        },
    )
