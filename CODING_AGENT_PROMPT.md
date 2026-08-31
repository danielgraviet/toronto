# Paste this to your coding agent

Implement the project described in this directory.

Start by reading `AGENTS.md`, then `README.md`, then `docs/architecture.md`.

Build in the phase order in `AGENTS.md`. Do not skip the grader harness to chase UI.

Constraints recap:
- Trainer on Daytona snapshot `daytona-gpu` (H100; torch/vllm/trl already installed).
- Rewards from CPU Daytona sandbox pool only.
- Stage profile: ~32 sandboxes, short completions, 4–6 visible GRPO steps.
- Stage laptop is UI + operator controls.
- Ship ghost replay failsafe.

When done, update README with exact talk-day commands and mark `docs/IMPLEMENTATION_CHECKLIST.md`.
