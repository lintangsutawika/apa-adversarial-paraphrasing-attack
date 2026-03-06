import hydra

from rllm.agents.math_agent import MathAgent
from rllm.data.dataset import DatasetRegistry
from rllm.environments.base.single_turn_env import SingleTurnEnvironment

from rllm.trainer.agent_trainer import AgentTrainer

from src.adversarial_reward import adversarial_reward_fn
from src.prepare_gsm8k_data import prepare_gsm8k_data

@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):

    train_dataset, test_dataset = prepare_gsm8k_data()

    env_args = {"reward_fn": adversarial_reward_fn}

    trainer = AgentTrainer(
        agent_class=MathAgent,
        agent_args={},
        env_args=env_args,
        env_class=SingleTurnEnvironment,
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
