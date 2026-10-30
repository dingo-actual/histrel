# Histrel

A code agent with a single reasoning state and persistent tool-call history,
plus the benchmark harness used to measure it on SWE-bench.

---

> ## ⚠️ RESEARCH ARTIFACT — SANDBOX USE ONLY
>
> **The sole purpose of this agent is to demonstrate and benchmark the
> context-management and tool-use techniques applied to it. It is not a
> product and it is not hardened for real use.**
>
> **It absolutely must not be run outside a disposable sandbox environment.**
>
> Specifically, when you run it:
>
> - The `bash` tool executes arbitrary shell commands, with no approval
>   prompt, no allow-list, and no sandboxing of its own.
> - The `edit` tool writes to any path inside the workspace without asking for
>   confirmation.
> - The loop runs unattended for up to hundreds of turns and a multi-million
>   token budget, so a confused model has many chances to do damage.
> - Source code, files, and credentials reachable from the workspace are sent
>   to a third-party model provider.
>
> Run it inside a container (the bundled harness does this), a throwaway VM,
> or an equivalent isolation boundary — never on a workstation, a shared host,
> or any machine holding data you cannot afford to lose or leak.

---

## What this is

Histrel is split into two applications that share one package:

| Component | Path | Role |
| --- | --- | --- |
| Core agent | `src/histrel` | The agent loop, context management, and tool set. Runs as `python -m histrel`. |
| Benchmark harness | `benchmark` | Builds SWE-bench instance images, runs the agent in a container per instance, collects patches and usage. |

## Techniques demonstrated

- **Single reasoning state.** In `stateless` mode (the default) the agent does
  not replay a growing chat transcript. Each iteration renders one fresh user
  message containing the task, the recent observations, and the current
  repository state, so the prompt is bounded by design rather than by luck.
  `react` mode is retained as a baseline that replays the full history.
- **Duplicate-call guard.** Repeated reads, directory listings, and searches are
  detected against filesystem snapshots (path keys plus content digests) within
  a five-call window, and the repeat is answered with an explanation instead of
  another identical result.
- **SEARCH/REPLACE patching.** Edits are applied by a content-addressable patch
  engine with a three-tier match (exact, normalized, fuzzy) and indentation
  reconstruction, so a slightly mis-indented search block still lands correctly.
- **Repair routines.** Malformed tool-call JSON is reconstructed with one
  schema-constrained single-shot LLM call instead of failing the turn.

## Repository layout

```
src/histrel/
  cli.py, __main__.py      CLI entry point (python -m histrel)
  core/
    config.py              AgentConfig and TOML loading
    runtime/               Agent loop, tool execution, finalization
    journal.py             Append-only message journal
    duplicate_guard/       Repeat-call detection
  context/                 Provider message building, compaction, tokens, costs
  model/                   OpenAI-compatible adapter, prompts, routines
  tools/                   read/list/search/glob/edit/bash + patch engine
  logging/, metrics/       Loguru sinks and metrics

benchmark/
  bench_settings.toml      Harness settings (models, budgets, dataset)
  predict/                 Prediction runner, Docker orchestration, CLI
  common/                  Shared IO, schemas, logging, progress
  tests/                   Harness tests
```

## Requirements

- Python 3.11 (the project pins `>=3.11,<3.12`)
- [`uv`](https://docs.astral.sh/uv/) for environment management
- Docker (benchmark harness only)
- `ripgrep` and `fd` on `PATH` for the `search` and `glob` tools
- A Fireworks API key for real model calls

## Setup

```bash
uv sync
source .venv/bin/activate
```

Export the model key (the default config reads `FIREWORKS_API_KEY`):

```bash
export FIREWORKS_API_KEY="fw-..."
```

## Configuration

The canonical agent config is `histrel.toml` at the repository root. The
harness copies it per instance and fills in the placeholder values.

| Key | Meaning |
| --- | --- |
| `schema_version` | Config schema version (currently `1`). |
| `session_id`, `task_id` | Identifiers, overridden by the harness. |
| `mode` | `stateless` (per-iteration message) or `react` (replay history). |
| `agent_system_prompt` | `{ initialize = ..., step = ... }` system prompt pair. |
| `model_provider`, `model_name` | Provider id and model id. |
| `model_top_p`, `model_reasoning_effort` | Sampling and reasoning settings. |
| `model_context_limit_tokens`, `model_max_tokens_output` | Context window and output cap. |
| `model_api_key_env` | Environment variable holding the API key. |
| `[budgets]` | `max_wall_seconds`, `max_turns`, `max_tool_calls` (0 = unlimited). |
| `[compaction]` | Trigger fraction, target fraction, retained tail count. |
| `duplicate_call_window` | How many LLM calls a repeat read/list/search stays detectable (default 5). |

Harness settings live in `benchmark/bench_settings.toml`: model ids, total
token budget, dataset id and split, instance cap, per-instance timeout, Docker
base image, cancellation server address, and verbose flag.

## Running the agent directly

```bash
python -m histrel <config_path> <workspace_root> <output_dir> <user_msg>
```

Example, against a scratch copy of a repository:

```bash
python -m histrel \
  histrel.toml \
  /tmp/sandbox/repo \
  /tmp/sandbox/output \
  "Fix the failing test in tests/test_math.py"
```

Exit status is `0` for a successful session and `1` for any other outcome
(`failed`, `budget_exhausted`, `cancelled`, `infrastructure_error`).

## Running the SWE-bench benchmark

```bash
python -m benchmark.predict.cli --config benchmark/bench_settings.toml
```

Common options:

```bash
python -m benchmark.predict.cli \
  --run-name my-run \
  --split test \
  --dataset-id princeton-nlp/SWE-bench_Verified \
  --instance-ids django__django-11099,astropy__astropy-12907 \
  --timeout-seconds 7200 \
  --rebuild \
  --verbose                       # stream container stdout/stderr to stderr
```

The harness builds one image per instance, mounts the instance data read-only
and the output directory read-write, runs the agent inside the container, and
extracts the resulting patch. Runs are resumable: existing predictions are
skipped. A local HTTP server on `127.0.0.1:8765` accepts `POST /cancel` to stop
the active container. `--dry-run` writes a `PLAN.md` summary of the resolved
run instead of building or executing anything.

## Tools

| Tool | Purpose |
| --- | --- |
| `read` | Read a file, optionally a line range. |
| `edit` | Apply SEARCH/REPLACE patches; empty search creates a file. |
| `search` | Content search with ripgrep. |
| `list_dir` | List one directory level. |
| `glob` | Find files with fd. |
| `bash` | Run a shell command in the workspace. |

## Outputs

Each session writes artifacts to `<output_dir>`:

`manifest.resolved.json`, `messages.jsonl`, `token_usage_calls.jsonl`,
`costs.json`, `timings.json`, `tool_metrics.json`, `outcome.json`,
`result_manifest.json`, and `repository.diff`. Two more carry recorded session
facts: `compactions.jsonl` (one record per context compaction, with the
compacted and retained message counts and the summary) and
`source_context.json` (the session and task ids, loop mode, model, budgets,
workspace revision, and tool set that shaped every model call).

The benchmark writes to `<output_dir>/<run_name>/`:

- `bench_agent_predictions.jsonl` — one prediction record per instance
- `bench_agent_usage.jsonl` — token usage per instance
- `artifacts/<instance>/` — per-instance session artifacts

## Development

```bash
python -m pytest tests benchmark/tests          # default suite
python -m pytest tests benchmark/tests --all    # + slow, live-LLM, external-tool tests
python -m ruff format src/histrel benchmark tests
python -m ruff check src/histrel benchmark tests
python -m pyright src/histrel benchmark tests
python -m mypy src/histrel benchmark tests
```

Tests avoid mocking the code under test: fixtures supply real files, real
subprocesses, and Docker client stubs that record calls instead of contacting a
daemon.

## Cost note

The checked-in price catalog
(`src/histrel/context/fireworks_pricing_catalog.json`) is a last-resort
fallback. Prices are fetched live per model and cached; when both the fetch and
the cache miss, the catalog entry is used. Verify its numbers after changing
`model_name`, or reported costs will be wrong for the new model.

## License

Licensed under the [Apache License, Version 2.0](LICENSE).

Copyright 2026 Ryan P. Taylor — ryan@beta-reduce.net

Apache 2.0 grants a patent license and permits commercial and private use,
modification, and redistribution, provided the license and copyright notice are
kept and any modified files carry prominent change notices. It is not a
permissive-without-conditions license: it offers no warranty and no liability.
