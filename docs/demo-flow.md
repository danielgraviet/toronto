# Demo flow (15:00) — product ↔ talk mapping

Talk title working name: **Teach It in Public: 500 Tiny Universes**

This maps speaker beats to system calls. Full verbal outline lives with the speaker; this is the engineering companion.

| Time | Beat | System |
|------|------|--------|
| 0:00–1:30 | Hook; show 1 tile → grid | UI idle animation or pre-warmed grid heartbeat |
| 1:30–3:30 | Explain GENERATE→EVALUATE→REWARD→UPDATE | Static diagram overlay optional; grid visible |
| 3:30–5:00 | Audience picks puzzle + knobs | Operator: `POST /task/lock`, `POST /reward/knobs` |
| 5:00–6:30 | Baseline ("before") | `POST /baseline` → mostly red grid; pin one funny failure |
| 6:30–11:00 | Live teach | `POST /train` steps=1 repeated; pause once to change knobs mid-run |
| 11:00–13:00 | After photo | Side-by-side baseline vs latest best; curve final point |
| 13:00–15:00 | Punchline + handoff to next speaker | Idle grid; stop training |

## Mid-talk knob flip

Around 8:30 speaker may increase `lambda_ban`. Implement so in-flight step finishes, then next step uses new knobs. Emit `log` event explaining the change for the UI banner.

## Failure behavior during beats

| Failure | Operator action |
|---------|-----------------|
| Train step hangs &gt; 90s | `/stop`, skip to checkpoint after, or ghost |
| Pool depleted | show warning; eval with remaining; don't crash UI |
| Wi-Fi dies | local ghost mode on laptop (bundle ghost+UI statically if possible) |

## What "success" looks like on screen

- Baseline pass rate clearly low (e.g. &lt; 30%)
- Final pass rate clearly higher (e.g. &gt; 60%) on **same** task
- At least one before/after code pair readable on projector
