# Implementation checklist

Copy/paste progress tracker for the coding agent.

## Phase 0 — Smoke
- [ ] GPU sandbox: `torch.cuda.get_device_name(0)` shows H100
- [ ] Imports: `trl`, `transformers`, `vllm` (or defer vllm), `daytona`
- [ ] Create + delete one CPU sandbox with API key

## Phase 1 — Harness
- [ ] Task YAML loader
- [ ] sanitize / banned / harness builder
- [ ] Local unit tests (no Daytona)
- [ ] FizzBuzz+ grades correctly against reference completions

## Phase 2 — Pool
- [ ] Warm N sandboxes
- [ ] Parallel eval of M completions
- [ ] Timeout/banned/error → stable EvalResult
- [ ] Cleanup

## Phase 3 — Train
- [ ] GRPO one step on pool=8
- [ ] metrics.jsonl written
- [ ] checkpoint saved
- [ ] mean reward moves over 4–6 steps on FizzBuzz+ in rehearsal

## Phase 4 — API/UI
- [ ] `/health` `/state` `/task/lock` `/reward/knobs` `/baseline` `/train` `/stop`
- [ ] Websocket events paint grid + curve
- [ ] Before/after code panes

## Phase 5 — Stage
- [ ] `stage` profile defaults
- [ ] warm_pool script
- [ ] ghost record + replay
- [ ] README talk-day runbook filled
- [ ] Two full rehearsals
