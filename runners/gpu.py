"""GPU/model smoke checks for the Daytona ``daytona-gpu`` sandbox."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any



DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DEFAULT_PROMPT = '''# FizzBuzz with a Daytona twist.
# For n > 0:
# - multiples of 3 and 5 -> "FizzBuzz"
# - multiples of 3 -> "Fizz"
# - multiples of 5 -> "Buzz"
# - multiples of 7 -> "Daytona" (checked after 3/5 rules)
# - else -> the number as a string
# Do not use eval/exec. Pure Python.

def fizzbuzz(n: int) -> str:
    """Return the FizzBuzz+ label for n."""
'''


@dataclass(frozen=True, slots=True)
class GPUInfo:
    available: bool
    name: str | None
    memory_gib: float | None
    capability: tuple[int, int] | None = None


def inspect_gpu() -> GPUInfo:
    """Inspect CUDA without hiding the reason a GPU check failed."""
    try:
        torch = importlib.import_module("torch")
    except ImportError as exc:
        raise RuntimeError("PyTorch is not installed in this environment") from exc

    if not torch.cuda.is_available():
        return GPUInfo(available=False, name=None, memory_gib=None)

    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    return GPUInfo(
        available=True,
        name=torch.cuda.get_device_name(device),
        memory_gib=properties.total_memory / (1024**3),
        capability=(properties.major, properties.minor),
    )


def check_imports() -> dict[str, str]:
    """Return versions for the libraries needed by the GPU path."""
    versions: dict[str, str] = {}
    for module_name in ("torch", "transformers", "trl", "daytona"):
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise RuntimeError(f"Required module is unavailable: {module_name}") from exc
        versions[module_name] = str(getattr(module, "__version__", "unknown"))
    return versions


def load_model(model_name: str = DEFAULT_MODEL_NAME) -> tuple[Any, Any]:
    """Load a causal language model and tokenizer onto the available GPU."""
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    info = inspect_gpu()
    if not info.available:
        raise RuntimeError("CUDA is unavailable; run this inside daytona-gpu")

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
    )
    model.to("cuda")
    model.eval()
    return model, tokenizer


def generate_completions(
    model: Any,
    tokenizer: Any,
    prompt: str = DEFAULT_PROMPT,
    num_completions: int = 4,
    max_new_tokens: int = 192,
) -> list[str]:
    """Generate sampled completions; GRPO needs reward variation across samples."""
    if num_completions < 1:
        raise ValueError("num_completions must be positive")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        num_return_sequences=num_completions,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )
    prompt_length = inputs["input_ids"].shape[-1]
    return tokenizer.batch_decode(output_ids[:, prompt_length:], skip_special_tokens=True)


def inspect_trl() -> dict[str, str]:
    """Return the installed TRL trainer/config signatures for compatibility work."""
    import inspect as python_inspect

    trl = importlib.import_module("trl")
    return {
        "trl_version": str(getattr(trl, "__version__", "unknown")),
        "grpo_config": str(python_inspect.signature(trl.GRPOConfig)),
        "grpo_trainer": str(python_inspect.signature(trl.GRPOTrainer)),
    }


def validate_real_smoke_entrypoint() -> None:
    """Fail locally if the remote real-GRPO entrypoint is incomplete."""
    from trainer import smoke

    required = ("generate_completions", "load_model", "DaytonaGraderPool", "DaytonaReward")
    missing = [name for name in required if not callable(getattr(smoke, name, None))]
    if missing:
        raise RuntimeError(
            "trainer.smoke is missing runtime dependencies: " + ", ".join(missing)
        )


def run_synthetic_grpo(model_name: str = DEFAULT_MODEL_NAME, output_dir: str = "/tmp/toronto-grpo-smoke") -> None:
    """Run one GRPO step without Daytona to isolate the TRL training path."""
    torch = importlib.import_module("torch")
    datasets = importlib.import_module("datasets")
    trl = importlib.import_module("trl")

    model, tokenizer = load_model(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = datasets.Dataset.from_dict({"prompt": [DEFAULT_PROMPT]})

    def synthetic_reward(completions: list[Any], **_: Any) -> list[float]:
        rewards = []
        for completion in completions:
            if isinstance(completion, list):
                completion = completion[0]["content"]
            rewards.append(1.0 if "return" in str(completion) else -1.0)
        print(f"Synthetic rewards: {rewards}")
        return rewards

    config = trl.GRPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_generations=2,
        generation_batch_size=2,
        max_completion_length=32,
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
        reward_funcs=synthetic_reward,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(output_dir)
    print(f"Synthetic GRPO smoke complete: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the Toronto GPU/model path")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--profile", default="stage", choices=("stage", "full"))
    parser.add_argument("--num-completions", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--remote", action="store_true", help="run this smoke check in Daytona")
    parser.add_argument("--gpu-type", default="RTX-PRO-6000")
    parser.add_argument(
        "--show-build-logs", action="store_true", help="stream Daytona image build logs"
    )
    parser.add_argument("--keep", action="store_true", help="keep the remote sandbox after the run")
    parser.add_argument("--inspect-trl", action="store_true", help="print installed TRL signatures")
    parser.add_argument("--synthetic-grpo", action="store_true", help="run one GRPO step without Daytona")
    parser.add_argument("--real-grpo-smoke", action="store_true", help="run one GRPO step with Daytona rewards")
    parser.add_argument("--sweep", action="store_true", help="run the independent GRPO parameter sweep")
    parser.add_argument("--pool-size", type=int, default=None, help="CPU grader sandboxes for real GRPO")
    parser.add_argument("--train-steps", type=int, default=None, help="GRPO optimizer steps")
    parser.add_argument("--eval-samples", type=int, default=None, help="baseline/final samples")
    parser.add_argument("--seed", type=int, default=42, help="baseline/final sampling seed")
    parser.add_argument("--task-id", default="fizzbuzz_plus", help="task YAML id")
    parser.add_argument("--learning-rate", type=float, default=5e-6, help="GRPO learning rate")
    parser.add_argument("--train-batch-size", type=int, default=None, help="prompts per GRPO update")
    parser.add_argument("--max-completion-length", type=int, default=None)
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args()
    from trainer.config import get_profile

    profile = get_profile(args.profile)
    args.pool_size = args.pool_size or profile.pool_size
    args.train_steps = args.train_steps or profile.train_steps
    args.eval_samples = args.eval_samples or profile.eval_samples
    args.train_batch_size = args.train_batch_size or profile.train_batch_size
    args.max_completion_length = args.max_completion_length or profile.max_completion_length

    if args.remote:
        asyncio.run(run_remote(args))
        return

    info = inspect_gpu()
    print(f"GPU available: {info.available}")
    print(f"GPU name: {info.name or 'none'}")
    print(f"GPU memory GiB: {info.memory_gib or 'unknown'}")
    print(f"Compute capability: {info.capability or 'unknown'}")
    print(f"Imports: {check_imports()}")
    if args.inspect_trl:
        print(f"TRL API: {inspect_trl()}")
        return
    if args.synthetic_grpo:
        run_synthetic_grpo(args.model)
        return
    if args.real_grpo_smoke or args.sweep:
        raise RuntimeError("--real-grpo-smoke and --sweep must be used with --remote")

    model, tokenizer = load_model(args.model)
    completions = generate_completions(
        model, tokenizer, num_completions=args.num_completions, max_new_tokens=args.max_new_tokens
    )
    for index, completion in enumerate(completions, start=1):
        print(f"\n--- completion {index} ---\n{completion}")


async def run_remote(args: argparse.Namespace) -> None:
    """Provision a GPU sandbox and execute this file inside it."""
    if args.real_grpo_smoke or args.sweep:
        validate_real_smoke_entrypoint()
    from .daytona import DaytonaRunner, SandboxSpec, build_gpu_image

    runner = DaytonaRunner()
    sandbox = None
    try:
        sandbox = await runner.create(
            SandboxSpec(
                image=build_gpu_image(),
                name=f"toronto-gpu-smoke-{uuid.uuid4().hex[:8]}",
                gpu=1,
                gpu_type=args.gpu_type,
                ephemeral=True,
                timeout_seconds=1800,
                show_build_logs=args.show_build_logs,
            )
        )
        print(f"Created remote sandbox: {getattr(sandbox, 'id', 'unknown')}")
        local_file = __file__
        remote_file = "/tmp/toronto_gpu.py"
        await runner.upload(sandbox, local_file, remote_file)
        command = (
            f"python {remote_file} --num-completions {args.num_completions} "
            f"--max-new-tokens {args.max_new_tokens}"
        )
        if args.inspect_trl:
            command += " --inspect-trl"
        if args.synthetic_grpo:
            command += " --synthetic-grpo"
        if args.real_grpo_smoke or args.sweep:
            remote_root = "/tmp/toronto"
            await runner.upload_tree(sandbox, Path(__file__).parents[1], remote_root)
        if args.sweep:
                command = (
                    f"cd {remote_root} && PYTHONPATH={remote_root} python -m trainer.sweep "
                    f"--pool-size {args.pool_size} --eval-samples {args.eval_samples} "
                    f"--seed {args.seed} --task-id {args.task_id} "
                    f"--output-root /tmp/toronto-grpo-sweep"
                )
        else:
            command = (
                f"cd {remote_root} && PYTHONPATH={remote_root} python -m trainer.smoke "
                f"--pool-size {args.pool_size} --train-steps {args.train_steps} "
                f"--eval-samples {args.eval_samples} --seed {args.seed} "
                f"--task-id {args.task_id} --learning-rate {args.learning_rate} "
                f"--train-batch-size {args.train_batch_size} "
                f"--max-completion-length {args.max_completion_length}"
            )
            if args.baseline_only:
                command += " --baseline-only"
        remote_env = {}
        for name in ("DAYTONA_API_KEY", "HF_TOKEN"):
            if value := os.getenv(name):
                remote_env[name] = value
        response = await runner.exec(
            sandbox, command, timeout_seconds=900, env=remote_env or None
        )
        output = getattr(response, "result", None)
        if output is None:
            artifacts = getattr(response, "artifacts", None)
            output = getattr(artifacts, "stdout", None) if artifacts else None
        print(output if output is not None else response)
        if getattr(response, "exit_code", 0) not in (0, None):
            raise RuntimeError(f"Remote GPU smoke command failed with exit code {response.exit_code}")
    finally:
        if sandbox is not None and args.keep:
            print("Keeping remote sandbox")
        elif sandbox is not None:
            await runner.delete(sandbox)
            print("Deleted remote sandbox")
        await runner.close()


if __name__ == "__main__":
    main()
