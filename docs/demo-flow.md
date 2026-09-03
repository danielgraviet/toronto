# Demo flow (15:00) · product ↔ talk mapping

Talk title: **Teach It in Public**

Speaker notes live in [`talk-script-15m.md`](talk-script-15m.md). Slides live in [`../talk/index.html`](../talk/index.html). This file is the engineering companion.

Talk day training path is **HF**: `make demo`.

| Time | Beat | System |
|------|------|--------|
| 0:00–1:30 | Hook. Finetuning can be approachable on Daytona. | Slides (Go live). CLI idle. |
| 1:30–3:00 | RL intuition and Toronto thread | Slides |
| 3:00–5:30 | Five words as separate visual slides | Slides 3 to 7 |
| 5:30–8:30 | Human GRPO with Two Sum (four approaches, live scores) | Slide 8. **Around 7:00 start `make demo`.** |
| 8:30–10:30 | Daytona stack plus wider primitive | Slides 9 and 10. Narrate spin-up if CLI is still provisioning. |
| 10:30–13:30 | Live CLI window | Rich demo UI. Baseline, training steps, finale. |
| 13:30–15:00 | Mantras and questions | Keep finale on screen. |

## Why start the demo early

GPU provision, model load, and pool warm often take **1 to 3+ minutes** before baseline. Measured HF phases after rollouts begin are about **56 seconds** total. Start `make demo` near minute seven so spin-up overlaps the audience game and the stack section.

## Mid-talk cues

| Clock | Action |
|------:|--------|
| ~7:00 | `make demo` on the second display |
| ~10:30 | Turn the room to the CLI |
| >90s hung step | Stop. Ghost replay or prior finale. |

## Failure behavior

| Failure | Operator action |
|---------|-----------------|
| Train step hangs past about 90s | Stop. Skip to ghost or a saved finale. |
| Pool depleted | Warn. Grade with remaining sandboxes. Keep the UI alive. |
| Wi-Fi dies | Local ghost mode on the laptop. |

## What success looks like on screen

- Baseline pass rate clearly lower than the trained policy on the same task
- At least one readable before and after completion
- Audience can repeat collect → reward → update
