import hydra

from functools import partial

from rllm.agents.math_agent import MathAgent
from rllm.data.dataset import DatasetRegistry
from rllm.environments.base.single_turn_env import SingleTurnEnvironment

from rllm.trainer.agent_trainer import AgentTrainer

from src.adversarial_reward import adversarial_reward_fn
from src.agent import AdversarialAgent

@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):

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
