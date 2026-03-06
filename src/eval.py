import json
import os
from tqdm import tqdm

from vllm import SamplingParams, LLM

from src.prepare_gsm8k_data import prepare_gsm8k_data
from src.adversarial_reward import adversarial_reward_fn

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True, help="The model name or path to load the model from.")
    parser.add_argument("--temperature", type=float, default=0.7, help="The temperature to use for sampling.")
    parser.add_argument("--top_k", type=int, default=50, help="The top-k value to use for sampling.")
    parser.add_argument("--top_p", type=float, default=0.95, help="The top-p value to use for sampling.")
    parser.add_argument("--max_new_tokens", type=int, default=4192, help="The maximum number of new tokens to generate.")
    parser.add_argument("--output_dir", type=str, default="eval_results", help="Directory to save evaluation results.")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to evaluate (None for all).")
    args = parser.parse_args()

    # Prepare datasets
    train_dataset, test_dataset = prepare_gsm8k_data()

    # Initialize vLLM
    llm = LLM(model=args.model_name_or_path)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
    )

    # Prepare test samples
    if args.num_samples is not None:
        test_samples = list(test_dataset)[:args.num_samples]
    else:
        test_samples = list(test_dataset)

    print(f"Evaluating on {len(test_samples)} samples...")

    # Generate paraphrased questions
    prompts = [sample["question"] for sample in test_samples]
    outputs = llm.generate(prompts, sampling_params)

    # Evaluate each generated paraphrase
    results = []
    total_reward = 0.0
    success_count = 0

    for i, (sample, output) in enumerate(tqdm(zip(test_samples, outputs), total=len(test_samples), desc="Evaluating")):
        generated_text = output.outputs[0].text

        task_info = {
            "data_source": sample["data_source"],
            "ground_truth": sample["ground_truth"],
        }

        reward_output = adversarial_reward_fn(task_info, generated_text)

        result = {
            "idx": i,
            "original_question": sample["question"],
            "generated_response": generated_text,
            "ground_truth": sample["ground_truth"],
            "reward": reward_output.reward,
            "metadata": reward_output.metadata,
        }
        results.append(result)

        total_reward += reward_output.reward
        if reward_output.reward > 0:
            success_count += 1

    # Compute metrics
    avg_reward = total_reward / len(test_samples) if test_samples else 0.0
    success_rate = success_count / len(test_samples) if test_samples else 0.0

    print(f"\n=== Evaluation Results ===")
    print(f"Total samples: {len(test_samples)}")
    print(f"Successful adversarial examples: {success_count}")
    print(f"Success rate: {success_rate:.2%}")
    print(f"Average reward: {avg_reward:.4f}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "eval_results.json")

    summary = {
        "model": args.model_name_or_path,
        "num_samples": len(test_samples),
        "success_count": success_count,
        "success_rate": success_rate,
        "avg_reward": avg_reward,
        "config": {
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
        },
        "results": results,
    }

    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_file}")

