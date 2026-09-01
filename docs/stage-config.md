# Stage vs full config

## Profiles

| Knob | `stage` (talk) | `full` (offline confidence) |
|------|----------------|------------------------------|
| `SANDBOX_POOL_SIZE` | 16 (reliable minimum) | 32–128 |
| Completions / step | = pool size | = pool size |
| Live GRPO steps shown | 4 | 8+ |
| `max_completion_length` | 192 | 256–512 |
| Tasks active | 1 | 1–2 |
| UI ghost failsafe | required | optional |
| Model | 0.5B–1.7B base | up to guide default |

## Talk narrative numbers

Even on stage, **say**:

> "Production training uses hundreds of sandboxes per step. You're seeing the shape with thirty-two."

Never claim the on-screen grid is 500 if it isn't.

## Reward knobs (audience)

| Knob | Default stage | English |
|------|---------------|---------|
| pass rate | always on | "fraction of unit tests passed" |
| `lambda_len` | 0.0 or 0.1 | "tax long solutions" |
| `lambda_ban` | 2.0 when enabled | "fine banned patterns hard" |
| `lambda_speed` | 0.0 | optional; often off for clarity |

Speaker enables at most **two** extras live.

## Pre-stage checklist (T-30)

- [ ] `daytona-gpu` Active; CUDA visible
- [ ] Model weights cached on disk (no HF download during talk)
- [ ] CPU pool warmed; `/health` shows healthy ≥ 32
- [ ] UI open fullscreen; font large enough for back row
- [ ] Baseline dry-run once (discard metrics)
- [ ] Ghost run file present; switch tested
- [ ] Network path laptop → API verified on venue Wi-Fi (or phone tether backup)

## Executable stage profile

Run the capacity check before doors:

    uv run python -m scripts.warm_pool --profile stage

Then run the fixed winner configuration:

    uv run python -m runners.gpu --remote --real-grpo-smoke \
      --profile stage --gpu-type RTX-PRO-6000 --task-id two_sum_plus

The profile intentionally runs one configuration only. It is designed for
roughly 1–3 minutes after the GPU image and model cache are warm. The warm
script is a capacity probe, not a persistent pool: the trainer creates its own
pool so ownership and cleanup stay in one process.

## Resource notes for H100 box

- 100 GiB RAM is ample for small/medium base models + vLLM colocate.
- Keep other jobs off this sandbox during the talk.
- Disk 400 GiB: fine for a few checkpoints; clean old runs before show.
