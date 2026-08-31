# Trainer (GPU sandbox)

## Host

- Daytona snapshot: **`daytona-gpu`**
- Resources: **1× NVIDIA H100**, 16 vCPU, 100 GiB RAM, 400 GiB disk, region `us`
- Image already includes CUDA 13 toolkit, PyTorch 2.11 (cu130), vLLM 0.21, FlashInfer caches, TRL 1.5, Transformers 5.9, PEFT, bitsandbytes, accelerate, datasets

Entrypoint is `sleep infinity` — treat it as a long-lived machine. Install project code into a working dir (git clone or sync).

## Role

Own the learning loop:

1. Load weak base model (+ tokenizer).
2. Hold GRPO config for `stage` or `full` profile.
3. Call generator to produce completions.
4. Call grader pool via `reward_func`.
5. Apply GRPO update.
6. Publish metrics/checkpoints/events.

## Model strategy

**Goal:** visible before→after in 4–6 steps, not SOTA.

Suggested order to try:

1. `Qwen/Qwen2.5-0.5B` or similar small **base** model (fast iterate on H100)
2. Bump to `Qwen/Qwen2.5-1.5B` or `Qwen/Qwen3-1.7B-Base` if lift is too weak
3. Optional: start from `checkpoint-0` saved after 0 steps of a prior rehearsal (guarantees ugly before)

Prefer **base** over instruct so completions look like code-continuations (matches Daytona guide prompts).

## GRPO / TRL notes

Installed: `trl==1.5.0`. APIs may differ slightly from older blog snippets — **read the installed TRL GRPOTrainer signature inside the sandbox** before copying guide code blindly.

Patterns to preserve from the Daytona guide:

- `GRPOConfig` with `logging_steps=1`
- `num_generations` / batch sizing such that completions per step ≈ pool size
- Dedicated asyncio event loop; `run_async()` inside sync `reward_func`
- `max_completion_length` short on stage (128–256)
- Save `metrics.jsonl` and checkpoints under `OUTPUT_DIR`

### Stage profile (defaults)

```text
pool_size = 32
per_step_completions = 32
max_steps_live = 6
max_prompt_length = 256
max_completion_length = 192
learning_rate = ~1e-5 to 8e-6 (tune in rehearsal)
bf16 = True
```

Exact GRPOConfig fields must match TRL 1.5 — implementer verifies.

### Full profile

Closer to the public guide (hundreds of sandboxes). Not used on stage. Keep as config flag for offline confidence runs.

## Generation

**Preferred:** vLLM colocated (already installed, FlashInfer warmed) for speed.

**Fallback:** Hugging Face `generate` if vLLM+TRL integration fights pins. For a 15-minute talk, a slower generate with reliable training beats a fragile vLLM path.

Document which path is default in README after Phase 3.

## Reward integration

```python
def reward_func(prompts, completions, **kwargs):
    stats = run_async(evaluate_batch(pool, completions, prompts, knobs))
    return [to_scalar(s, knobs) for s in stats]
```

`to_scalar`:

```text
if error or timeout or banned:
    return -1.0
reward = passed / total
reward -= knobs.lambda_len * norm_lines(completion)
reward -= knobs.lambda_ban * banned_count_or_zero
reward += knobs.lambda_speed * speed_term  # optional; default 0
clip or leave raw — pick one and test
```

Banned detection should usually short-circuit **before** sandbox exec (as in the guide). Still safe if exec happens — sandbox is isolated.

## Control surface hooks

Trainer module should expose (Python API used by FastAPI):

- `lock_task(task_id)`
- `set_knobs(lambda_len, lambda_ban, lambda_speed)`
- `run_baseline(num_samples)` — generate+eval, **no** optimizer step
- `train(num_steps)`
- `get_state()` — task, knobs, step, last metrics, best completion
- `load_checkpoint(path)` / `save_checkpoint()`
- `shutdown()` — stop train, cleanup pool

## Performance budget

Target wall clock per step on stage: **≤ 60s**, hard uncomfortable at **90s**.

If over budget, cut in order:

1. completions 32 → 16
2. max_completion_length ↓
3. enable faster generate path
4. reduce visible steps to 4 and lean on strong before/after narrative

## Operator commands (talk day)

Pseudo:

```bash
# inside daytona-gpu
export DAYTONA_API_KEY=...
export STAGE_PROFILE=stage
python -m api.main           # serves UI + control
# separate terminal if needed
python -m scripts.warm_pool --n 32
```

Exact module paths left to implementer; keep them boring.
