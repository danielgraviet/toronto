# Risks and failsafe

## Top risks

1. **Step too slow** — GRPO step &gt; 90s kills energy.
2. **Venue network** — laptop cannot reach GPU sandbox URL.
3. **Model too strong or too weak** — no visible lift / always green.
4. **SDK/API drift** — daytona or trl call shapes differ from docs.
5. **Pool create throttling** — warming 32 sandboxes fails partially.
6. **Live demo hubris** — debugging on stage.

## Mitigations

| Risk | Mitigation |
|------|------------|
| Slow step | Stage profile cuts; HF vs vLLM bakeoff in rehearsal; fewer tokens |
| Network | Preview URL + tether backup; **ghost mode offline on laptop** |
| Lift | Tune puzzle hardness + banned patterns; choose weaker start checkpoint |
| API drift | Thin adapters; smoke tests; pin versions already on snapshot |
| Pool | Warm early; allow start with ≥16 with banner; retry create |
| Hubris | Rehearse twice with ghost fallback muscle memory |

## Ghost run spec

Record during a good rehearsal:

- Event jsonl identical to websocket payload shapes
- Wall-clock timestamps relative to t0
- Optional: speaker notes markers (`hook`, `baseline`, `train`, `after`)

UI button **GHOST** switches input to file. No training process required.

Also keep a silent screen capture as last-resort video under `ghost/backup.mp4` (not required for v1 code).

## Explicit non-claims on stage

- Don't claim training is CPU-only.
- Don't claim 500 sandboxes if showing 32.
- Don't claim the model is production-grade after 5 steps.
- Do claim: verifiable rewards + isolated parallel eval is the Daytona-shaped idea.
