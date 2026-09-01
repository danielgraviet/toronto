# Teach It in Public: 500 Tiny Universes

Live demo + talk system for a **15-minute** Toronto AI Engineering talk (Daytona.io).

The audience co-designs a reward for a coding puzzle. A **weak model** is GRPO-finetuned live. Completions are graded in parallel **CPU Daytona sandboxes**. The trainer runs on a **Daytona GPU sandbox** (H100).

## One-sentence architecture

> **GPU sandbox = generate + GRPO update. CPU sandbox pool = unit-test graders. Stage laptop = UI + votes only.**

## Repo layout (target)

```
toronto-talk-demo/
├── README.md                 # this file
├── AGENTS.md                 # coding-agent handoff (start here)
├── docs/
│   ├── architecture.md       # system design, data flow, services
│   ├── demo-flow.md          # 15-min beat sheet + stage UI states
│   ├── stage-config.md       # stage vs production knobs
│   ├── trainer.md            # GRPO / TRL / vLLM on daytona-gpu
│   ├── graders.md            # CPU sandbox pool + test harness
│   ├── control-api.md        # trainer ↔ stage UI protocol
│   └── risks-and-failsafe.md # timing, network, ghost run
├── tasks/                    # puzzle definitions (prompts, tests, bans)
│   ├── README.md
│   ├── fizzbuzz_plus.yaml
│   ├── two_sum.yaml
│   └── safe_parser.yaml
└── (implementation to be built by coding agent)
    ├── trainer/              # GRPO loop on GPU sandbox
    ├── graders/              # Daytona CPU pool + harness
    ├── api/                  # control + event websocket
    ├── ui/                   # stage display
    └── scripts/              # warm pool, ghost record, smoke tests
```

## Success criteria

1. Cold start ("before") pass rate is visibly low on the chosen puzzle.
2. After 4–6 GRPO steps, pass rate is visibly higher on the **same** prompts.
3. Audience can pick puzzle + toggle 1–2 reward knobs without breaking the run.
4. Sandbox grid + reward curve update live; one step ideally &lt; 60s wall clock (hard cap ~90s).
5. If the live train dies, operator can cut to a pre-recorded ghost run and keep narrating.

## Non-goals

- Not a full reproduction of the 500-sandbox blog training job on stage.
- Not multi-GPU, not multi-task live, not instruct-tuning UX polish.
- Not enterprise data agents (that's another speaker's talk).

## Upstream reference

Daytona guide (shape to follow for harness + pool + reward):

- https://www.daytona.io/docs/en/guides/reinforcement-learning/trl-grpo-training/

Official example uses Qwen3-1.7B-Base + 500 sandboxes + vLLM colocated on one big GPU. **This project shrinks that to stage size** and splits UI onto a separate client.

## Hardware already available

| Role | Where | Specs |
|------|--------|--------|
| Trainer | Daytona snapshot `daytona-gpu` | 1× NVIDIA H100, 16 vCPU, 100 GiB RAM, 400 GiB disk, `us` |
| Graders | Daytona CPU sandboxes | Default sandbox image; pool of 32 (stage) |
| Stage UI | Speaker laptop browser | Talks to control API over network |

### `daytona-gpu` image highlights (already baked)

- CUDA 13 toolkit (`CUDA_HOME=/usr/local/cuda`)
- `torch==2.11.0` (cu130), `vllm==0.21.0`, FlashInfer cubins cached
- `trl==1.5.0`, `transformers==5.9.0`, `peft`, `bitsandbytes`, `accelerate`, `datasets`
- Entrypoint: `sleep infinity` (long-lived system sandbox)

Snapshot name: `daytona-gpu` · Region: `us`

## Quick mental model for implementers

```
Audience vote → lock task + λ knobs
     → Trainer.generate(batch)
     → CPU pool.evaluate(completions) → rewards
     → GRPOTrainer.step
     → stream metrics/grid/best-code → UI
```

## Stage rehearsal

The stage profile uses one short run instead of a parameter sweep:

~~~
uv run python -m scripts.warm_pool --profile stage
uv run python -m runners.gpu --remote --real-grpo-smoke \
  --profile stage --gpu-type RTX-PRO-6000 --task-id two_sum_plus
~~~

It uses 16 CPU graders, 4 GRPO steps, 16 baseline/holdout samples, and a
192-token completion limit. The warm command verifies Daytona capacity and
releases its probe sandboxes; the training process creates and owns the pool
used for grading. Keep a successful rehearsal event log as the offline ghost
fallback.

## Frontend rehearsal

The static stage UI works without the control API and simulates the complete
story so buttons can be rehearsed offline:

    uv run python -m http.server 4173 --directory ui

Open http://localhost:4173 and use Run baseline, Train 4 steps, or Ghost
replay. To point the same UI at a future control API, open it with an api
query parameter, for example:

    http://localhost:4173/?api=http://localhost:8080

For real Daytona execution, use two terminals:

    make backend
    make ui

Then open http://localhost:4173/?api=http://localhost:8080. The API process
starts the real GPU runner; the browser never receives Daytona or HF secrets.

## License / talk credit

Internal demo for Daytona talk by Daniel (Thi) Graviet. Not a public product.
