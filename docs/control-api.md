# Control API + Stage UI protocol

## Purpose

Let the speaker laptop drive the trainer without SSH heroics during the talk.

## Transport

- **HTTP JSON** for commands (idempotent where possible)
- **WebSocket** for high-frequency events (grid, curve, logs)

Base URL: trainer host inside `daytona-gpu`, exposed via Daytona preview URL or SSH tunnel.

## HTTP endpoints (suggested)

| Method | Path | Body | Effect |
|--------|------|------|--------|
| GET | `/health` | — | liveness, cuda ok, pool healthy count |
| GET | `/state` | — | full snapshot for UI refresh |
| GET | `/tasks` | — | list audience-facing tasks |
| POST | `/task/lock` | `{ "task_id": "fizzbuzz_plus" }` | lock puzzle |
| POST | `/reward/knobs` | `{ "lambda_len": 0.1, "lambda_ban": 2.0, "lambda_speed": 0 }` | set knobs |
| POST | `/baseline` | `{ "n": 32 }` | generate+eval, no weight update |
| POST | `/train` | `{ "steps": 1 }` | run N GRPO steps |
| POST | `/checkpoint/load` | `{ "path": "..." }` | load weights |
| POST | `/stop` | — | cancel in-flight work |
| POST | `/mode` | `{ "mode": "live" \| "ghost" }` | switch feed |

All mutations return `{ok, state}` or error `{ok:false, error}`.

## WebSocket events

Client connects to `/ws`. Server pushes:

- `hello` — version, profile, model name
- `state` — full state (also on connect)
- `sandbox_result` — tile update
- `step_started` / `step_finished`
- `curve_point` — `{step, mean_reward, pass_rate}`
- `best_completion` — `{step, code, reward}`
- `log` — operator-facing string
- `error` — recoverable/fatal flag

## UI layout (stage mode)

```
┌─────────────────┬────────────────────┬─────────────────┐
│ Prompt + best   │  Sandbox grid NxM  │ Reward curve    │
│ / before-after  │  green/red tiles   │ pass rate       │
│ code panes      │                    │ knobs readout   │
└─────────────────┴────────────────────┴─────────────────┘
│ Operator bar: lock task | knobs | baseline | train | ghost | stop │
```

Audience vote can be:

- Built-in big buttons (speaker clicks winner), or
- Integration with Slido (speaker manually locks result)

**v1:** speaker clicks winning puzzle + toggles knobs on operator bar. No mandatory third-party voting dependency.

## Auth

v1: unauthenticated local/preview URL. Optional shared bearer token via env `STAGE_TOKEN` if exposure is broader.

## Ghost mode

When `mode=ghost`, UI consumes `ghost/run.jsonl` event stream with original timing (or accelerated 1.5×). Trainer idle.
