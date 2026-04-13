import hydra
import os

import copy
from functools import partial

from rllm.agents.math_agent import MathAgent
from rllm.data.dataset import DatasetRegistry
from rllm.environments.base.single_turn_env import SingleTurnEnvironment

from rllm.trainer.agent_trainer import AgentTrainer

from src.adversarial_reward import adversarial_reward_fn
from src.agent import AdversarialAgent


def _patch_transformers_v5_compat() -> None:
    import transformers

    if hasattr(transformers, "AutoModelForVision2Seq"):
        return

    replacement = getattr(transformers, "AutoModelForImageTextToText", None)
    if replacement is None:
        raise ImportError(
            "Transformers v5 compatibility patch could not find AutoModelForImageTextToText."
        )

    transformers.AutoModelForVision2Seq = replacement


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
                raise ValueError(
                    "Expected each dataset entry to contain a non-empty 'messages' list."
                )
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


def _maybe_disable_verl_fsdp_sync_module_states() -> None:
    """Optionally skip FSDP module-state sync when all ranks already load weights.

    VERL's FSDP1 path constructs `FSDP(... sync_module_states=True)`, which triggers
    a rank-0 broadcast during init. On Babel, every rank is already reading the full
    checkpoint shards locally before that step, so for smoke-debugging we can
    disable the redundant sync via an explicit env flag and test whether the
    timeout is purely in that broadcast.
    """

    if os.environ.get("APA_DISABLE_FSDP_SYNC_MODULE_STATES") != "1":
        return

    import verl.workers.fsdp_workers as fsdp_workers

    if getattr(fsdp_workers, "_apa_sync_patch_applied", False):
        return

    original_build_model_optimizer = (
        fsdp_workers.ActorRolloutRefWorker._build_model_optimizer
    )

    def _patched_build_model_optimizer(self, *args, **kwargs):
        original_fsdp = fsdp_workers.FSDP

        def _patched_fsdp(*fsdp_args, **fsdp_kwargs):
            if fsdp_kwargs.get("sync_module_states"):
                if os.environ.get("RANK", "0") == "0":
                    print("APA patch: disabling FSDP sync_module_states during init")
                fsdp_kwargs["sync_module_states"] = False
            return original_fsdp(*fsdp_args, **fsdp_kwargs)

        fsdp_workers.FSDP = _patched_fsdp
        try:
            return original_build_model_optimizer(self, *args, **kwargs)
        finally:
            fsdp_workers.FSDP = original_fsdp

    fsdp_workers.ActorRolloutRefWorker._build_model_optimizer = (
        _patched_build_model_optimizer
    )
    fsdp_workers._apa_sync_patch_applied = True


@hydra.main(
    config_path="pkg://rllm.trainer.config",
    config_name="agent_ppo_trainer",
    version_base=None,
)
def main(config):
    _patch_transformers_v5_compat()
    _use_messages_as_verl_prompt()
    _maybe_disable_verl_fsdp_sync_module_states()

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
