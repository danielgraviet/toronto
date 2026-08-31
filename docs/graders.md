# Graders (CPU Daytona sandboxes)

## Role

Parallel, isolated execution of **unit-test harnesses** for model-written code. No GPUs. No model weights.

## Pool lifecycle

1. **Warm** before doors: create `N` sandboxes (`STAGE_PROFILE=stage` → 32).
2. **Reuse** across baseline + all GRPO steps.
3. **Cleanup** on process exit / emergency stop (best-effort delete).

Creation should be concurrent (`asyncio.gather`). Log failures per sandbox; refuse to start show if healthy pool &lt; `N/2` unless operator overrides.

## Per-completion pipeline

1. `sanitize_completion(raw)` → function body (indent-based for base-model completions; also handle fenced markdown if model emits it).
2. `has_banned_pattern(body, task)` → if hit, return failed stats immediately (reward -1), skip exec.
3. `build_test_harness(task, body)` → full Python script.
4. `sandbox.code_interpreter.run_code(code, timeout=...)` (or equivalent SDK call for installed daytona version).
5. Parse **last non-empty stdout line** as JSON: `{"results":[true,false,...]}`.
6. Return `EvalResult{no_error, num_passed, num_tests, duration_ms, sandbox_id, banned}`.

## Harness contract

Harness must:

- Define the function under test via prompt preamble + model body.
- Run deterministic tests (seed random if used).
- Compare against a reference implementation or gold outputs.
- Print a single JSON object on the last line.
- Catch per-test exceptions → `false` for that test, not crash the whole process when possible.

Timeouts / interpreter errors / empty stdout → `no_error=False`.

## Task schema

YAML in `tasks/`:

```yaml
id: fizzbuzz_plus
title: FizzBuzz+
func_name: fizzbuzz
prompt: |
  # ... completion-mode preamble ending with def line + docstring ...
banned_patterns:
  - "eval("
  - "exec("
  - "__import__"
tests:
  # represent however implementer prefers: list of input/output pairs
  # or executable snippets like the sorting guide
reference: |
  # reference implementation source, inlined into harness
audience_blurb: "Classic FizzBuzz with a twist the room can understand in 10 seconds."
```

Implementer may store tests as structured `{input, output}` and generate harness code — clearer than the guide's "tests as Python expression strings." Either is fine if reliable.

## Parallelism

- One in-flight eval per sandbox.
- Round-robin or free-list scheduling.
- If completions &gt; pool size, queue (stage profile should avoid this by matching sizes).

## Safety

- Never pass host secrets into harness.
- No network needed for puzzles; if sandbox has network, banned patterns should include obvious exfil helpers for the "safe parser" task.
- Keep `MAX_TIMEOUT_SECONDS` at 1s for stage puzzles (tune if flaky).

## UI events

For each completion emit:

```json
{
  "type": "sandbox_result",
  "step": 3,
  "index": 12,
  "sandbox_id": "...",
  "passed": 4,
  "total": 5,
  "banned": false,
  "error": false,
  "duration_ms": 220
}
```

UI maps this to green/yellow/red tiles.
