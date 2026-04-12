import hydra

import copy
from functools import partial

from rllm.agents.math_agent import MathAgent
from rllm.data.dataset import DatasetRegistry
from rllm.environments.base.single_turn_env import SingleTurnEnvironment

from rllm.trainer.agent_trainer import AgentTrainer

from src.adversarial_reward import adversarial_reward_fn
from src.agent import AdversarialAgent


def _use_messages_as_verl_prompt() -> None:
    """Keep VERL parquet prompts aligned with the real task messages.

    rllm's default DatasetRegistry VERL postprocessing writes a placeholder
    prompt and moves the real task into extra_info. That makes the tokenizer
    path and environment path see different inputs. For this project we want the
    VERL prompt payload to be the actual chat messages we train on.
    """

    def _apply_verl_postprocessing(cls, data):
        processed_data = []
        for entry in data:
            messages = entry.get("messages")
            if hasattr(messages, "tolist"):
                messages = messages.tolist()
            messages = copy.deepcopy(messages)
            if not isinstance(messages, list) or not messages:
                raise ValueError("Expected each dataset entry to contain a non-empty 'messages' list.")
            processed_data.append(
                {
                    "prompt": messages,
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": None,
                    },
                    "extra_info": entry,
                }
            )
        return processed_data

    DatasetRegistry.apply_verl_postprocessing = classmethod(_apply_verl_postprocessing)


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    _use_messages_as_verl_prompt()

    if "task_name" not in config:
        task_name = "gsm8k"
    else:
        task_name = config["task_name"]

    if task_name == "gsm8k":
        print("Using GSM8K task")
        from src.tasks.gsm8k import prepare_gsm8k_data
        from rllm.rewards.reward_fn import math_reward_fn

        reward_fn = partial(adversarial_reward_fn, task_reward_fn=math_reward_fn)
        train_dataset, test_dataset = prepare_gsm8k_data()
    elif task_name == "mbpp":
        print("Using MBPP task")
        from src.tasks.mbpp import prepare_mbpp_data, mbpp_reward_fn

        reward_fn = partial(adversarial_reward_fn, task_reward_fn=mbpp_reward_fn)
        train_dataset, test_dataset = prepare_mbpp_data()

    trainer = AgentTrainer(
        agent_class=AdversarialAgent,
        agent_args={},
        env_args={"reward_fn": reward_fn},
        env_class=SingleTurnEnvironment,
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
