import hydra

from rllm.agents.math_agent import MathAgent
from rllm.data.dataset import DatasetRegistry
from rllm.environments.base.single_turn_env import SingleTurnEnvironment

from rllm.trainer.agent_trainer import AgentTrainer

from src.adversarial_reward import adversarial_reward_fn

@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):

    if "task_name" not in config:
        config.task_name = "gsm8k"

    if config.task_name == "gsm8k":
        from src.tasks.gsm8k import prepare_gsm8k
        train_dataset, test_dataset = prepare_gsm8k()
    elif config.task_name == "mbpp":
        from src.tasks.mbpp import prepare_mbpp
        train_dataset, test_dataset = prepare_mbpp()

    victim_model = config.get("victim_model", "litellm_proxy/azure_ai/gpt-oss-120b")
    reference_model = config.get("reference_model", "litellm_proxy/azure_ai/Llama-3.3-70B-Instruct")

    # Pass victim and reference model info to the dataset
    train_dataset = train_dataset.map(lambda x: {"victim_model": victim_model, "reference_model": reference_model})
    test_dataset = test_dataset.map(lambda x: {"victim_model": victim_model, "reference_model": reference_model})

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
