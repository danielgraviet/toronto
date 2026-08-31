# Tasks (coding puzzles)

Each task is a YAML file consumed by the trainer/grader.

## Required fields

- `id`, `title`, `func_name`, `prompt`, `banned_patterns`, `tests`, `reference`, `audience_blurb`

## Design rules

1. Explainable in &lt; 20 seconds to mixed audience.
2. Unit tests must be deterministic and fast (&lt; 1s total).
3. Include at least one "gotcha" test so weak models fail baseline.
4. Banned patterns encode the audience reward story (no `eval`, no cheating builtins).
5. Prompt style: **completion mode** for base models (preamble ends at `def ...:` / docstring).

## Stage default

If voting is slow, lock **`fizzbuzz_plus`**.

## Files

- `fizzbuzz_plus.yaml` — default
- `two_sum.yaml` — slightly more "engineer"
- `safe_parser.yaml` — security-flavored bridge (no stealing the OWASP skills talk)
