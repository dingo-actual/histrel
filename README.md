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
  repository state.
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
git clone https://github.com/dingo-actual/histrel.git
cd histrel
git submodule update --init --recursive
uv sync --group benchmark
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
| `agent_system_prompt` | Placeholder pair; the CLI renders the real prompts. |
| `model_provider`, `model_name` | Provider id and model id. |
| `model_top_p`, `model_reasoning_effort` | Sampling and reasoning settings. |
| `model_context_limit_tokens`, `model_max_tokens_output` | Context window and output cap. |
| `model_api_key_env` | Environment variable holding the API key. |
| `[budgets]` | `max_wall_seconds`, `max_turns`, `max_tool_calls` ( = unlimited). |
| `[compaction]` | Trigger fraction, target fraction, retained tail count. |
| `duplicate_call_window` | How many LLM calls a repeat read/list/search stays detectable (default 5). |

Harness settings live in `benchmark/bench_settings.toml`: model ids, total
token budget, dataset id and split, instance cap, per-instance timeout, Docker
base image, and verbose flag.

## Running the agent directly

```bash
python -m histrel [--mode stateless|react] <config_path> <workspace_root> <output_dir> <user_msg>
```

`--mode` selects the agent loop: `stateless` (the default, a per-iteration
user message) or `react` (replays the full history).

Example, against a scratch copy of a repository:

```bash
python -m histrel \
  --mode stateless \
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
  --mode stateless \
  --split test \
  --dataset-id princeton-nlp/SWE-bench_Verified \
  --instance-ids django__django-11099,astropy__astropy-12907 \
  --timeout-seconds 72 \
  --repeat 3 \
  --rebuild \
  --verbose                       # stream container stdout/stderr to stderr
```

`--mode` (default `stateless`) is passed through to every agent container as
`python -m histrel --mode ...`.

The harness builds one image per instance, mounts an allowlisted task record
read-only and the output directory read-write, runs the agent inside the
container, and extracts the resulting patch. Reference patches and evaluation
metadata never leave the host. Runs are resumable: existing predictions are
skipped. `--dry-run` writes a `PLAN.md` summary of the resolved run instead of
building or executing anything.

### Grading repeated runs

SWE-bench turns a predictions file into a dictionary keyed by `instance_id`, so
the merged `bench_agent_predictions.jsonl` cannot be graded as repetitions.
That merged file is an analysis artifact, not a grading input.

`benchmark.score.cli` grades each `repeat_NN/bench_agent_predictions.jsonl` in
its own SWE-bench evaluation run, with run ID `<run-name>-repeat-NN`, then
aggregates the reports:

```bash
python -m benchmark.score.cli benchmark/results/<run-name>
```

The dataset and split default to the values recorded in the run metadata.
`--force` re-grades a repetition whose report already exists; without it a
batch resumes where it stopped.

`resolution_aggregate.json` holds the resolve rate of each repetition, the
mean, median, min, and max resolve rate across repetitions, and per-instance
resolution counts. The rates are `null` unless every scheduled repetition has
a report: a repetition whose evaluation never ran is unobserved, not failed.

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

| File | Contents |
| --- | --- |
| `manifest.resolved.json` | The fully resolved config used for the session. |
| `messages.jsonl` | The model message journal, one record per message. |
| `token_usage_calls.jsonl` | One record per model call: token counts, call kind, provider request id. |
| `costs.json` | Token and cost totals computed from the price catalog. |
| `timings.json` | Wall-clock seconds per timing category, plus a metadata block. |
| `tool_metrics.json` | Per-tool call metrics. |
| `source_context.json` | Fixed session inputs: session and task ids, loop mode, model, budgets, workspace revision, tool set. |
| `outcome.json` | Final session status and its reason. |
| `repository.diff` | Workspace diff after the session. |
| `compactions.jsonl` | One record per context compaction: compacted and retained message counts, and the summary. |
| `result_manifest.json` | Hashes of every other artifact; written last. |

The benchmark writes to `<output_dir>/<run_name>/`:

| Path | Contents |
| --- | --- |
| `bench_agent_predictions.jsonl` | Analysis artifact, not a grading input: every prediction from every repetition, tagged with a `repetition` number. Grade the per-repetition file instead. |
| `bench_agent_usage.jsonl` | Token usage per instance. |
| `bench_agent_usage_aggregate.json` | Written only with `--repeat`: per-instance and overall sample count, sum, mean, and median of each usage metric |
| `artifacts/<instance>/` | Per-instance session artifacts. |
| `instance_mounts/<instance>/instance.json` | Task record mounted read-only at `/workspace/instance`: instance id, repo, base commit, problem statement, hints, and test-runner flag only. |
| `resolution_aggregate.json` | Written by `benchmark.score.cli`: per-repetition resolve rate, statistics across repetitions, and per-instance resolution counts. |
| `repeat_NN/` | Written only with `--repeat`: one full run directory per repetition, with the same files, including its own gradeable `bench_agent_predictions.jsonl`. |

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

## Benchmark Results

The following are results from five independent runs of ten SWE-bench Verified instances, each run using Histrel and a comparable ReAct agent loop. All results used GLM 5.3 Flash.

### ReAct

| SWE-bench Instance ID | Successes | Success Rate | Input Tokens (Median) | Input Tokens (Mean) | Max Context (Median) | Max Context (Mean) | Cache Hit Rate (Median) | Cache Hit Rate (Mean) | Cost (Median) | Cost (Mean) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| astropy__astropy-7671 | 5 | 1 | 348616 | 36766 | 16246 | 15634 | 0.9462 | 0.9468 | 0.0227 | 0.0203 |
| django__django-10554 |  |  | 2986489 | 2981368 | 52183 | 52585.2 | 0.9814 | 0.9817 | 0.106 | 0.1075 |
| matplotlib__matplotlib-26208 | 3 | 0.6 | 2954209 | 2283069 | 43369 | 38442.8 | 0.9827 | 0.9797 | 0.1054 | 0.0839 |
| mwaskom__seaborn-3069 | 4 | 0.8 | 2970349 | 2849816 | 44805 | 45112.8 | 0.9841 | 0.9823 | 0.105 | 0.1052 |
| pallets__flask-5014 | 5 | 1 | 228443 | 19738 | 9539 | 9416 | 0.9448 | 0.9334 | 0.0112 | 0.0128 |
| pydata__xarray-3993 | 4 | 0.8 | 170388 | 2073256 | 37926 | 37506.6 | 0.98 | 0.9763 | 0.0876 | 0.0888 |
| pytest-dev__pytest-7324 | 4 | 0.8 | 2660347 | 2205884 | 39579 | 34572.8 | 0.9807 | 0.9789 | 0.1119 | 0.0992 |
| scikit-learn__scikit-learn-13496 | 5 | 1 | 28436 | 38684 | 15791 | 17244.2 | 0.9458 | 0.9444 | 0.0277 | 0.0268 |
| sphinx-doc__sphinx-9461 |  |  | 2950393 | 2919105 | 43217 | 43286.2 | 0.9843 | 0.9842 | 0.108 | 0.1128 |
| sympy__sympy-13878 | 1 | 0.2 | 2968382 | 2941952 | 49441 | 49549 | 0.9793 | 0.978 | 0.1154 | 0.128 |
| Total |  |  | 2831635.5 | 1920632.94 | 4018 | 34334 | 0.9686 | 0.9797 | 0.1045 | 0.0785 |

### Histrel

| SWE-bench Instance ID | Successes | Success Rate | Input Tokens (Median) | Input Tokens (Mean) | Max Context (Median) | Max Context (Mean) | Cache Hit Rate (Median) | Cache Hit Rate (Mean) | Cost (Median) | Cost (Mean) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| astropy__astropy-7671 | 5 | 1 | 21667 | 25206 | 16187 | 16458 | 0.7294 | 0.7372 | 0.029 | 0.0257 |
| django__django-10554 | 4 | 0.8 | 1803241 | 1998559 | 49833 | 52831.8 | 0.898 | 0.8973 | 0.1463 | 0.161 |
| matplotlib__matplotlib-26208 | 4 | 0.8 | 137546 | 1205231.4 | 38435 | 36181 | 0.8602 | 0.8658 | 0.0865 | 0.0884 |
| mwaskom__seaborn-3069 | 5 | 1 | 1344172 | 1183206 | 34232 | 35770.4 | 0.8753 | 0.8619 | 0.0824 | 0.0751 |
| pallets__flask-5014 | 5 | 1 | 172148 | 169275.2 | 14226 | 13166.2 | 0.6816 | 0.6712 | 0.0215 | 0.0194 |
| pydata__xarray-3993 | 2 | 0.4 | 738147 | 673001.4 | 26613 | 24988.6 | 0.8375 | 0.8245 | 0.0515 | 0.0446 |
| pytest-dev__pytest-7324 | 4 | 0.8 | 267375 | 296781.6 | 18786 | 18607.6 | 0.7554 | 0.7655 | 0.0321 | 0.0333 |
| scikit-learn__scikit-learn-13496 | 5 | 1 | 550489 | 515055.2 | 23089 | 22210.4 | 0.8345 | 0.7885 | 0.0446 | 0.0338 |
| sphinx-doc__sphinx-9461 |  |  | 1380733 | 1350423.8 | 47129 | 44833 | 0.8812 | 0.8721 | 0.0975 | 0.0925 |
| sympy__sympy-13878 | 5 | 1 | 2860098 | 2307876.6 | 57824 | 54626 | 0.9153 | 0.8645 | 0.1683 | 0.1462 |
| Total |  |  | 711927.5 | 995147.02 | 2763 | 31967.3 | 0.8148 | 0.8395 | 0.0502 | 0.072 |

### Difference

| SWE-bench Instance ID | Successes | Success Rate | Input Tokens (Median) | Input Tokens (Mean) | Max Context (Median) | Max Context (Mean) | Cache Hit Rate (Median) | Cache Hit Rate (Mean) | Cost (Median) | Cost (Mean) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| astropy__astropy-7671 |  |  | 62.15 % | 68.56 % | 99.64 % | 105.27 % | 77.09 % | 77.86 % | 127.91 % | 126.88 % |
| django__django-10554 | 4 | 0.8 | 60.38 % | 67.03 % | 95.5 % | 100.47 % | 91.5 % | 91.4 % | 138.1 % | 149.74 % |
| matplotlib__matplotlib-26208 | 1 | 0.2 | 46.56 % | 52.79 % | 88.62 % | 94.12 % | 87.54 % | 88.37 % | 82.01 % | 105.41 % |
| mwaskom__seaborn-3069 | 1 | 0.2 | 45.25 % | 41.52 % | 76.4 % | 79.29 % | 88.95 % | 87.74 % | 78.54 % | 71.44 % |
| pallets__flask-5014 |  |  | 75.36 % | 85.76 % | 149.14 % | 139.83 % | 72.14 % | 71.91 % | 192.12 % | 151.6 % |
| pydata__xarray-3993 | -2 | -0.4 | 43.32 % | 32.46 % | 70.17 % | 66.62 % | 85.46 % | 84.45 % | 58.81 % | 50.25 % |
| pytest-dev__pytest-7324 |  |  | 10.05 % | 13.45 % | 47.46 % | 53.82 % | 77.03 % | 78.2 % | 28.71 % | 33.56 % |
| scikit-learn__scikit-learn-13496 |  |  | 193.59 % | 133.14 % | 146.22 % | 128.8 % | 88.24 % | 83.49 % | 161.12 % | 126.41 % |
| sphinx-doc__sphinx-9461 |  |  | 46.8 % | 46.26 % | 109.05 % | 103.57 % | 89.52 % | 88.61 % | 90.28 % | 82.08 % |
| sympy__sympy-13878 | 4 | 0.8 | 96.35 % | 78.45 % | 116.96 % | 110.25 % | 93.46 % | 88.39 % | 145.77 % | 114.21 % |
| Total |  |  | 25.14 % | 51.81 % | 68.77 % | 93.11 % | 84.12 % | 85.69 % | 48.04 % | 91.72 % |

## License

Licensed under the [Apache License, Version 2.0](LICENSE).

Copyright 2026 Ryan P. Taylor — ryan@beta-reduce.net

Apache 2 grants a patent license and permits commercial and private use,
modification, and redistribution, provided the license and copyright notice are
kept and any modified files carry prominent change notices. It is not a
permissive-without-conditions license: it offers no warranty and no liability.
