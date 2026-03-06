import os
import re

from rllm.rewards.reward_types import RewardOutput
from rllm.rewards.reward_fn import math_reward_fn

import litellm

kwargs = {
    "base_url": os.environ.get("LLM_API_URL", "http://localhost:8000"),
    "api_key": os.environ.get("LLM_API_KEY", "fake_api_key_for_testing"),
}

def adversarial_reward_fn(task_info: dict, action: str) -> RewardOutput:
    """
    A reward function for math tasks that implements the RewardFunction protocol.

    Args:
        task: The task dictionary containing data_source, ground_truth and other metadata
        action: The agent's response/solution

    Returns:
        float: The calculated reward value based on math evaluation
    """

    try:
        paraphrased_question = re.search(r"<problem>(.*?)</problem>", action, re.DOTALL).group(1).strip()
    except:
        paraphrased_question = None

    if paraphrased_question is None:
        return RewardOutput(reward=0.0, metadata={"error": "Invalid action format"})

    messages = [
        {"role": "system", "content": "Reason step by step and put your final answer within \\boxed{}."},
        {"role": "user", "content": paraphrased_question},
        ]
    
    try:
        victim_response = litellm.completion(
            model="litellm_proxy/azure_ai/gpt-oss-120b",
            messages=messages,
            **kwargs
        ).choices[0].message.content.strip()
        reference_response = litellm.completion(
            model="litellm_proxy/azure_ai/Llama-3.3-70B-Instruct",
            messages=messages,
            **kwargs
        ).choices[0].message.content.strip()

    except Exception as e:
        victim_response = None
        reference_response = None

    victim_reward = math_reward_fn(task_info, victim_response)
    reference_reward = math_reward_fn(task_info, reference_response)

    if victim_reward.reward == 0 and reference_reward.reward == 1:
        reward = 1.0
    else:
        reward = 0.0

    return RewardOutput(reward=reward, metadata={
        "paraphrased_question": paraphrased_question,
        "victim_response": victim_response,
        "reference_response": reference_response,
        "victim_reward": victim_reward.reward,
        "reference_reward": reference_reward.reward,
    })
