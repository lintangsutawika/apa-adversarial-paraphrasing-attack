# Adversarial Paraphrasing Attack

## Train

You will need to setup litellm_proxy variables.
```
export LLM_API_URL=...
export LLM_API_KEY=...
```

Or you can also change the models in `src/adversarial_reward.py`, I use gpt-oss-120b as the reference model and Llama-3.3-70B-Instruct as the victim model.

To train
```
bash scripts/train_gsm8k.sh
```

## Evaluate

```
uv run --isolated src/eval.py \
    --model_name_or_path neulab/adversarial-paraphraser-qwen3-8b \
    --output_dir ./eval_results/ \
    --num_samples 100
```