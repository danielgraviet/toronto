# Stage vs full config

## Profiles

| Knob | `stage` (talk) | `full` (offline confidence) |
|------|----------------|------------------------------|
| `SANDBOX_POOL_SIZE` | 32 | 128–500 |
| Completions / step | = pool size | = pool size |
| Live GRPO steps shown | 4–6 | 8+ |
| `max_completion_length` | 128–256 | 512 |
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

## Resource notes for H100 box

- 100 GiB RAM is ample for small/medium base models + vLLM colocate.
- Keep other jobs off this sandbox during the talk.
- Disk 400 GiB: fine for a few checkpoints; clean old runs before show.
