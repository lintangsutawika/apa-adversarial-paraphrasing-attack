import os
import re
from typing import Callable

from rllm.rewards.reward_types import RewardOutput

import litellm

kwargs = {
    "base_url": os.environ.get("LLM_API_URL", "http://localhost:8000"),
    "api_key": os.environ.get("LLM_API_KEY", "fake_api_key_for_testing"),
}


def _completion_content(model: str, messages: list[dict]) -> str | None:
    try:
        return (
            litellm.completion(model=model, messages=messages, **kwargs)
            .choices[0]
            .message.content.strip()
        )
    except Exception:
        return None


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

    match = re.search(r"<problem>(.*?)</problem>", action, re.DOTALL)
    paraphrased_question = match.group(1).strip() if match else None

    if paraphrased_question is None:
        return RewardOutput(reward=0.0, metadata={"error": "Invalid action format"})

    messages = [dict(message) for message in task_info["target_prompts"]]
    messages[-1]["content"] = messages[-1]["content"].replace(
        "__PARAPHRASED_QUESTION__", paraphrased_question
    )

    victim_model = task_info.get(
        "victim_model", "litellm_proxy/azure_ai/Llama-3.3-70B-Instruct"
    )
    reference_models = task_info.get(
        "reference_models",
        task_info.get("reference_model", "litellm_proxy/azure_ai/gpt-oss-120b"),
    )
    if isinstance(reference_models, str):
        reference_models = [reference_models]
    if not reference_models:
        reference_models = ["litellm_proxy/azure_ai/gpt-oss-120b"]

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
