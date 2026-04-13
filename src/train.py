import hydra
import os
from pathlib import Path

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


def _patch_uv_cached_verl_model_import() -> None:
    uv_cache_dir = os.environ.get("UV_CACHE_DIR")
    if not uv_cache_dir:
        return

    patch_snippet = (
        "import transformers as _apa_transformers\n"
        'if not hasattr(_apa_transformers, "AutoModelForVision2Seq") and '
        'hasattr(_apa_transformers, "AutoModelForImageTextToText"):\n'
        "    _apa_transformers.AutoModelForVision2Seq = "
        "_apa_transformers.AutoModelForImageTextToText\n\n"
    )

    cache_root = Path(uv_cache_dir)
    for candidate in cache_root.rglob("verl/utils/model.py"):
        try:
            text = candidate.read_text()
        except OSError:
            continue

        if "AutoModelForVision2Seq" not in text:
            continue
        if "_apa_transformers.AutoModelForVision2Seq" in text:
            continue

        marker = "from transformers import ("
        if marker not in text:
            continue

        candidate.write_text(text.replace(marker, patch_snippet + marker, 1))


def _patch_ray_worker_setup_hook() -> None:
    import rllm.trainer.verl.ray_runtime_env as ray_runtime_env

    env_vars = ray_runtime_env.PPO_RAY_RUNTIME_ENV.setdefault("env_vars", {})
    repo_root = os.path.dirname(os.path.dirname(__file__))
    env_vars["PYTHONPATH"] = f"{repo_root}/src:{repo_root}"
    ray_runtime_env.PPO_RAY_RUNTIME_ENV["worker_process_setup_hook"] = (
        "ray_worker_setup.setup"
    )


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
    _patch_uv_cached_verl_model_import()
    _patch_ray_worker_setup_hook()
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
