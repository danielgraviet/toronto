"""One real GRPO step using the Daytona CPU reward path."""

from __future__ import annotations

import asyncio
import importlib

from graders.pool import DaytonaGraderPool
from runners.gpu import DEFAULT_MODEL_NAME, load_model
from tasks import TaskLoader

from .reward import DaytonaReward


def run(model_name: str, pool_size: int, output_dir: str) -> None:
    torch = importlib.import_module("torch")
    datasets = importlib.import_module("datasets")
    trl = importlib.import_module("trl")

    model, tokenizer = load_model(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    task = TaskLoader().load("fizzbuzz_plus")
    dataset = datasets.Dataset.from_dict({"prompt": [task.prompt]})
    pool = DaytonaGraderPool(requested_size=pool_size)
    reward = DaytonaReward(pool)
    try:
        config = trl.GRPOConfig(
            output_dir=output_dir,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            num_generations=2,
            generation_batch_size=2,
            max_completion_length=64,
            learning_rate=1e-5,
            max_steps=1,
            logging_steps=1,
            report_to="none",
            bf16=torch.cuda.is_available(),
            use_vllm=False,
            gradient_checkpointing=True,
        )
        trainer = trl.GRPOTrainer(
            model=model,
            reward_funcs=reward,
            args=config,
            train_dataset=dataset,
            processing_class=tokenizer,
        )
        trainer.train()
        trainer.save_model(output_dir)
        print(f"Real Daytona GRPO smoke complete: {output_dir}")
    finally:
        reward.close()
        asyncio.run(pool.close())


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run one GRPO step with real Daytona rewards")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--pool-size", type=int, default=2)
    parser.add_argument("--output-dir", default="/tmp/toronto-real-grpo-smoke")
    args = parser.parse_args()
    run(args.model, args.pool_size, args.output_dir)


if __name__ == "__main__":
    main()
