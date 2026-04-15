# Environments

## Current Best-Match Branch/Env Pair

- Branch baseline: `bbf0c1c028480bee6a3fb2ab45ceb38eec943373`
- Preferred Babel env: `/data/user_data/barryw/envs/apa_env`

Expected key package versions in `apa_env`:

- `python 3.13`
- `torch 2.8.0+cu128`
- `transformers 4.57.0`
- `vllm 0.11.0`

This branch is not a good fit for the newer `q35sys312` stack (`transformers 5.2.0`, `vllm 0.19.0`). That newer environment exposed additional incompatibilities in `verl`, `vllm`, and FSDP/offload behavior.

## What Broke On The Newer Stack

The main problem with `bbf0c1c` on `q35sys312` was not one single bug. It was a branch/runtime mismatch that exposed several independent incompatibilities:

- `transformers 5.2.0` vs this branch's Qwen3 path
  - Qwen3 model loading and chat-template behavior differ from the older `transformers 4.57.0` path this branch was developed against.
  - On the newer stack, Qwen3 also defaulted into FlashAttention-related paths that were not satisfied in the env until `flash_attn` was installed.
  - The generic `rllm.disable_thinking` flag was also not the correct control for Qwen3 chat-template behavior on this stack.

- `vllm 0.19.0` vs this branch's older rollout assumptions
  - On the newer stack we saw vLLM-specific failures during model/weight loading and sleep-wake behavior.
  - Example symptoms included CuMem / wake-up failures and later `CUDA error: invalid argument` paths during weight load or cache management.
  - The older `vllm 0.11.0` in `apa_env` is much closer to what this branch expects.

- `verl` / `rllm` API drift
  - The newer env surfaced protocol mismatches around `DataProto` vs `TensorDict` behavior, old/new field names, and actor/update assumptions.
  - Those failures were real, but many only appeared because we were trying to force an older branch through a newer trainer/runtime stack.

- FSDP / offload behavior drift
  - The old branch expects the older `fsdp2` path and its normalization/offload semantics.
  - On the newer env we hit actor CPU-offload assertions and other update-time failures that were not simple repo-code bugs.
  - In practice, `fsdp2` itself was not the issue; the issue was using it inside a mismatched newer surrounding stack.

- Ray / distributed launch behavior
  - On Babel, Ray's default GPU env rewriting caused duplicate-GPU / wrong-rank visibility problems on 8-GPU launches.
  - This was fixed with `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`.
  - This is separate from stale-state problems after failed launches.

- Stale distributed state after failures
  - After bad launches, the node could retain stale Ray workers, vLLM processes, GPU allocations, and process-store state.
  - Symptoms included NCCL/TCPStore nonce mismatches, actor creation failures, and hangs that later hit the default ~600 second NCCL watchdog timeout.
  - This is an operational cleanup issue, not a model-code incompatibility.

In short: the newer stack failures were real, but many of them were secondary effects of running an older branch against newer `transformers`, `vllm`, `verl`, and FSDP behavior than it was built for.

## Required Runtime Configuration

These settings were required to make the current code version run cleanly enough to debug training behavior on Babel:

- `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`
  - Avoids the duplicate-GPU / wrong-rank GPU visibility problem on 8-GPU launches.
  - Root cause: Ray was rewriting `CUDA_VISIBLE_DEVICES` for colocated worker processes in a way that made different ranks infer inconsistent device ownership.
- `LLM_API_URL=https://api.fireworks.ai/inference/v1`
- `LLM_API_KEY=<fireworks key>`
- `LLM_VICTIM_MODEL=fireworks_ai/accounts/fireworks/models/llama-v3p3-70b-instruct`
- `LLM_REFERENCE_MODEL=fireworks_ai/accounts/fireworks/models/qwen3-8b`

## Current Branch-Specific Launch Notes

- On 8 GPUs with the older `fsdp2` path, `actor_rollout_ref.actor.ppo_mini_batch_size=1` normalizes to `0` and crashes. Use `ppo_mini_batch_size=8` with `ppo_micro_batch_size_per_gpu=1`.
- The current smoke/debug target uses `data.max_prompt_length=2048` and `data.max_response_length=2048`.
- For Qwen3 on this branch/runtime, the trainer-path no-thinking knob that actually worked in live PPO was `rllm.disable_thinking=True`.
  - In the live 8B smoke run, this removed `<think>` from `Sample 0`, reduced response length from ~2048 to ~81 tokens, and changed all completions from `TRUNCATION` to `ENV_DONE`.
  - Do not use `/no_think` in the dataset prompt.
- `+data.apply_chat_template_kwargs.enable_thinking=False` still works in direct offline vLLM probes, but it was not sufficient in the live trainer integration path on this stack.
- Qwen3 non-thinking mode should use the model-card settings: `temperature=0.7`, `top_p=0.8`, `top_k=20`.
- Keep the GSM8K attack prompt minimal.
  - Do not add examples or hand-written transformation recipes to force specific rewrite styles.
- Launch from the repo root with `PYTHONPATH=/home/barryw/apa-bbf0c1c python -m src.train ...`.
  - This avoids the import-path ambiguity we saw when invoking `python /home/barryw/apa-bbf0c1c/src/train.py` directly.
- Reward/provider failures should fail loudly. Silent LiteLLM fallthroughs make broken provider calls look like legitimate zero-reward samples.

## Known Operational Issues

- Failed launches can leave stale Ray, vLLM, and GPU processes on the node.
  - Before reruns, kill stale workers, run `ray stop --force`, and verify GPUs are free.
- Earlier bad distributed launches sometimes hung long enough to hit the default NCCL watchdog timeout around 600 seconds.
  - Treat this as stale/distributed-state fallout, not as evidence that the current reward/training path is wrong.
- Cold-node startup can look hung before Hydra creates a new output directory or any GPU memory is allocated.
  - On `babel-q9-24`, we observed a fresh `python -m src.train ...` spend multiple minutes inside Hydra/`verl`/`transformers` import-time file scanning on the NFS-backed env.
  - Symptoms: parent `python -m src.train` process stays in `D` state with `wchan=filemap_update_page`, `read_bytes` keeps climbing, no new `outputs/...` run directory appears yet, and GPUs remain near `0 MiB`.
  - A direct `import src.train` and `prepare_gsm8k_data()` check succeeded on the same node, so this slow phase was cold filesystem/import startup rather than a broken branch/env pairing.
- On this `apa_env` stack, live 8B rollout samples still began with `<think>` even with `+data.apply_chat_template_kwargs.enable_thinking=False` present in the config dump.
  - We tested a repo-side `sitecustomize.py` workaround to forward chat-template kwargs into the fully async client, but it did not change the live behavior and was removed immediately.
  - Current conclusion: the active generation path for this trainer is still bypassing or overriding the expected Qwen3 hard no-thinking switch somewhere deeper in the stack.
- The Ray runtime environment in this path forwards only a whitelist of env-var prefixes and did not include `PYTHONPATH` or our custom patch env var.
  - That explains why the repo-side worker patch never reached the worker interpreters and was removed.
- Direct offline vLLM probe on `babel-q9-24` with `vllm 0.11.0` and `Qwen/Qwen3-8B`:
  - `LLM.chat(..., chat_template_kwargs={"enable_thinking": False})` worked as expected: no `<think>`, short output, valid `<problem>...</problem>` rewrite.
  - Raw token-ID generation also worked when the prompt IDs were built with `tokenizer.apply_chat_template(..., enable_thinking=False)` and passed to `LLM.generate(...)`.
  - `/no_think` without the hard switch was worse: it produced an empty `<think></think>` wrapper before the rewrite.
  - So the Qwen/vLLM mechanism itself works on this node; the failure is specific to the trainer integration path, not to Qwen3 or offline vLLM in general.
- Direct trainer-path probe with `rllm.disable_thinking=True` on `babel-q9-24`:
  - `Sample 0` showed an empty `<think></think>` assistant scaffold in the masked prompt, but the actual model output after the separator started directly with `<problem>...</problem>` and contained no `<think>`.
  - Response length dropped to ~81 tokens and all 8 trajectories finished with `ENV_DONE` instead of `TRUNCATION`.
  - Rewards were still all `0.0`, so this fixes the thinking/truncation failure mode but not the zero-reward attack-quality problem.
- We also tested a repo-side Ray worker hook that patched the two known loss points:
  - `rllm.experimental.fully_async.client.RolloutClient.chat_completion()`
  - `verl.workers.rollout.schemas.AsyncRolloutRequest._handle_apply_chat_template()`
  - The hook definitely loaded: `runtime_env_agent.log` showed `worker_process_setup_hook=src.worker_patches.setup`, packaged repo `working_dir`, and `APA_APPLY_CHAT_TEMPLATE_KWARGS={"enable_thinking": false}` in worker env.
  - Despite that, the first live trainer `Sample 0` still began with `<think>`, so this repo-side worker hook did not fix the actual generation path and was removed immediately.

## Data/Pipeline Notes

- `src/tasks/gsm8k.py` now forces `load_from_cache_file=False` during `datasets.map(...)`.
  - This is important while iterating on the GSM8K attack prompt, otherwise Hugging Face dataset caching can reuse stale preprocessing output.
- The task data is also registered into `~/.rllm/datasets/gsm8k/`.
  - Clearing that directory may still be useful between major prompt/data changes.
- Reward debugging note:
  - We verified with a direct one-sample reward probe that `reward=0.0` can be genuine even when provider calls are healthy.
  - In the validated probe, `completion_failed=False`, `victim_reward=1.0`, and `reference_rewards=[1.0]`, so all-zero rewards reflected weak attacks rather than broken reward plumbing.

## Current State Of The Training Problem

- The main remaining blocker is not environment stability.
- The live 1.7B path can run trajectories and training steps, but rewards are still often all `0`.
- When rewards are all `0`, PPO has no learning signal, so `pg_loss` and `grad_norm` staying near `0` is expected.
- The remaining work is to improve attacker behavior so rewards become non-zero, then carry the same clean path to 8B.
