# AGENTS.md — Coding agent handoff

You are implementing the **Toronto talk live demo**: GRPO finetuning of a weak code model on a Daytona **GPU** sandbox, with rewards from parallel Daytona **CPU** sandboxes, plus a stage UI.

Read in this order:

1. `README.md`
2. This file
3. `docs/architecture.md`
4. `docs/trainer.md` + `docs/graders.md` + `docs/control-api.md`
5. `docs/stage-config.md` + `docs/demo-flow.md` + `docs/risks-and-failsafe.md`
6. `tasks/*`

## Mission

Build a working vertical slice, then harden for stage:

1. **Grade path**: one puzzle harness runs in one CPU sandbox and returns structured pass/fail.
2. **Swarm path**: N CPU sandboxes grade N completions in parallel.
3. **Train path**: TRL GRPO on `daytona-gpu` (H100) uses that swarm as `reward_func`.
4. **Control path**: stage UI can lock task, set reward knobs, start/stop, and stream live state.
5. **Failsafe**: script to record a ghost run; UI can replay it.

## Hard constraints

- **CPU sandboxes never run the model.** They only execute test harnesses.
- **Trainer runs on Daytona snapshot `daytona-gpu`** (1× H100). Stack is preinstalled (torch cu130, vllm 0.21, trl 1.5, transformers 5.9, peft, bitsandbytes). Prefer using what's already there; pin versions if you add deps.
- **Stage defaults**: 32 sandboxes, 4–6 visible GRPO steps, max completion length 128–256, one live task.
- **Do not** require 500 sandboxes or an 80GB-only mental model on stage. Support a `STAGE` vs `FULL` config profile.
- Audience-facing reward must be explainable in one English sentence.
- Prefer boring, reliable code over clever abstractions.

## Preferred stack

| Layer | Choice |
|-------|--------|
| Trainer | Python 3, `trl` GRPOTrainer, `vllm` for generation if stable with installed pins; HF generate fallback OK |
| Graders | Daytona Python SDK (`AsyncDaytona`), `sandbox.code_interpreter.run_code` |
| Control API | FastAPI (or similar) on trainer host: REST for commands + WebSocket for events |
| UI | Simple web UI (Vite/React or plain HTML+JS). Full-screen stage mode. No login. |
| Config | YAML/JSON for tasks + env for secrets |
| Metrics | JSONL + websocket events; optional wandb later (not required for v1) |

## Environment variables (expected)

```bash
DAYTONA_API_KEY=...              # required
DAYTONA_API_URL=...              # if non-default
TRAINER_HOST=0.0.0.0
TRAINER_PORT=8080
STAGE_PROFILE=stage              # stage | full
MODEL_NAME=Qwen/Qwen2.5-0.5B     # start small; document how to bump to 1.7B
OUTPUT_DIR=./runs/live
SANDBOX_POOL_SIZE=32
```

Model choice is intentionally weak. Document final pick in README after smoke tests. Base models preferred over instruct so "before" looks worse.

## Implementation phases (do in order)

### Phase 0 — Smoke
- Create/delete one CPU sandbox.
- Confirm GPU sandbox sees CUDA: `torch.cuda.is_available()`, GPU name H100.
- Import `trl`, `vllm`, `daytona` successfully inside `daytona-gpu`.

### Phase 1 — Harness
- Implement `tasks/` loader.
- `sanitize_completion`, `has_banned_pattern`, `build_test_harness`, parse `{"results":[...]}`.
- Unit tests for sanitize/ban/reward math **without** Daytona.

### Phase 2 — Grader pool
- Pre-create pool of N sandboxes; reuse; cleanup on shutdown.
- Parallel evaluate completions; round-robin assignment.
- Timeouts → reward -1.0.

### Phase 3 — GRPO trainer
- Wire `reward_func(prompts, completions)` → async pool eval via event-loop bridge (same pattern as Daytona docs).
- Stage GRPOConfig: small batch aligned to pool size; `logging_steps=1`; short `max_steps`.
- Save checkpoints; emit per-step metrics (mean reward, pass rate, step time).

### Phase 4 — Control API + UI
- Commands: `lock_task`, `set_reward_knobs`, `run_baseline`, `train_steps`, `swap_checkpoint`, `emergency_stop`.
- Events: `sandbox_result`, `step_started`, `step_finished`, `best_completion`, `curve_point`, `error`.
- UI: left code / center grid / right curve; vote overlays.

### Phase 5 — Stage hardening
- Pool warm script (run before doors).
- Ghost recorder + replay.
- Kill-switch and degraded mode (eval-only swarm without weight update).

## Coding standards

- Typed Python where practical; no bare `except:`.
- Every Daytona call has timeout + structured error → reward failure, not trainer crash.
- Log sandbox id on failures.
- Do not commit API keys. Use `.env` (gitignored).
- Keep talk-specific copy out of core libs; put strings in UI/config.

## Out of scope for v1

- Multi-region scheduling, autoscaling pool, fancy auth, mobile UI, real-time collaborative editing, training on multiple GPUs.

## Definition of done (demo-ready)

- [ ] `scripts/smoke_grade.sh` passes
- [ ] `scripts/smoke_train_one_step.sh` completes one GRPO step with pool=8 on GPU sandbox
- [ ] `scripts/warm_pool.py --n 32` ready
- [ ] UI shows before→after lift on FizzBuzz+ in rehearsal
- [ ] Ghost replay works offline
- [ ] README "How to run on talk day" section filled with exact commands

## How to make decisions when unspecified

1. Follow Daytona GRPO guide patterns for harness/pool/reward.
2. Prefer stage reliability over matching blog hyperparameters.
3. If vLLM+TRL colocated is painful on these pins, ship HF generate first and leave vLLM as follow-up — **learning lift matters more than gen speed** for v1.
4. Ask the human only for: API keys, final model id if downloads fail, and public URL/port exposure for the stage laptop.
