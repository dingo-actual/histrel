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
| `[budgets]` | `max_wall_seconds`, `max_turns`, `max_tool_calls` (0 = unlimited). |
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
  --timeout-seconds 7200 \
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

The following are results from five independent runs of ten SWE-bench Verified instances, each run using Histrel and a comparable ReAct agent loop.

### ReAct

| SWE-bench Instance ID | Successes | Success Rate | Input Tokens (Median) | Input Tokens (Mean) | Max Context (Median) | Max Context (Mean) | Cache Hit Rate (Median) | Cache Hit Rate (Mean) | Cost (Median) | Cost (Mean) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| astropy__astropy-7671 | 5.0000 | 1.0000 | 348616.0000 | 367660.0000 | 16246.0000 | 15634.0000 | 0.9462 | 0.9468 | 0.0227 | 0.0203 |
| django__django-10554 | 0.0000 | 0.0000 | 2986489.0000 | 2981368.0000 | 52183.0000 | 52585.2000 | 0.9814 | 0.9817 | 0.1060 | 0.1075 |
| matplotlib__matplotlib-26208 | 3.0000 | 0.6000 | 2954209.0000 | 2283069.0000 | 43369.0000 | 38442.8000 | 0.9827 | 0.9797 | 0.1054 | 0.0839 |
| mwaskom__seaborn-3069 | 4.0000 | 0.8000 | 2970349.0000 | 2849816.0000 | 44805.0000 | 45112.8000 | 0.9841 | 0.9823 | 0.1050 | 0.1052 |
| pallets__flask-5014 | 5.0000 | 1.0000 | 228443.0000 | 197380.0000 | 9539.0000 | 9416.0000 | 0.9448 | 0.9334 | 0.0112 | 0.0128 |
| pydata__xarray-3993 | 4.0000 | 0.8000 | 1703880.0000 | 2073256.0000 | 37926.0000 | 37506.6000 | 0.9800 | 0.9763 | 0.0876 | 0.0888 |
| pytest-dev__pytest-7324 | 4.0000 | 0.8000 | 2660347.0000 | 2205884.0000 | 39579.0000 | 34572.8000 | 0.9807 | 0.9789 | 0.1119 | 0.0992 |
| scikit-learn__scikit-learn-13496 | 5.0000 | 1.0000 | 284360.0000 | 386840.0000 | 15791.0000 | 17244.2000 | 0.9458 | 0.9444 | 0.0277 | 0.0268 |
| sphinx-doc__sphinx-9461 | 0.0000 | 0.0000 | 2950393.0000 | 2919105.0000 | 43217.0000 | 43286.2000 | 0.9843 | 0.9842 | 0.1080 | 0.1128 |
| sympy__sympy-13878 | 1.0000 | 0.2000 | 2968382.0000 | 2941952.0000 | 49441.0000 | 49549.0000 | 0.9793 | 0.9780 | 0.1154 | 0.1280 |
| Total |  |  | 2831635.5000 | 1920632.9400 | 40180.0000 | 34334.0000 | 0.9686 | 0.9797 | 0.1045 | 0.0785 |

### Histrel

| SWE-bench Instance ID | Successes | Success Rate | Input Tokens (Median) | Input Tokens (Mean) | Max Context (Median) | Max Context (Mean) | Cache Hit Rate (Median) | Cache Hit Rate (Mean) | Cost (Median) | Cost (Mean) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| astropy__astropy-7671 | 5.0000 | 1.0000 | 216670.0000 | 252060.0000 | 16187.0000 | 16458.0000 | 0.7294 | 0.7372 | 0.0290 | 0.0257 |
| django__django-10554 | 4.0000 | 0.8000 | 1803241.0000 | 1998559.0000 | 49833.0000 | 52831.8000 | 0.8980 | 0.8973 | 0.1463 | 0.1610 |
| matplotlib__matplotlib-26208 | 4.0000 | 0.8000 | 1375460.0000 | 1205231.4000 | 38435.0000 | 36181.0000 | 0.8602 | 0.8658 | 0.0865 | 0.0884 |
| mwaskom__seaborn-3069 | 5.0000 | 1.0000 | 1344172.0000 | 1183206.0000 | 34232.0000 | 35770.4000 | 0.8753 | 0.8619 | 0.0824 | 0.0751 |
| pallets__flask-5014 | 5.0000 | 1.0000 | 172148.0000 | 169275.2000 | 14226.0000 | 13166.2000 | 0.6816 | 0.6712 | 0.0215 | 0.0194 |
| pydata__xarray-3993 | 2.0000 | 0.4000 | 738147.0000 | 673001.4000 | 26613.0000 | 24988.6000 | 0.8375 | 0.8245 | 0.0515 | 0.0446 |
| pytest-dev__pytest-7324 | 4.0000 | 0.8000 | 267375.0000 | 296781.6000 | 18786.0000 | 18607.6000 | 0.7554 | 0.7655 | 0.0321 | 0.0333 |
| scikit-learn__scikit-learn-13496 | 5.0000 | 1.0000 | 550489.0000 | 515055.2000 | 23089.0000 | 22210.4000 | 0.8345 | 0.7885 | 0.0446 | 0.0338 |
| sphinx-doc__sphinx-9461 | 0.0000 | 0.0000 | 1380733.0000 | 1350423.8000 | 47129.0000 | 44833.0000 | 0.8812 | 0.8721 | 0.0975 | 0.0925 |
| sympy__sympy-13878 | 5.0000 | 1.0000 | 2860098.0000 | 2307876.6000 | 57824.0000 | 54626.0000 | 0.9153 | 0.8645 | 0.1683 | 0.1462 |
| Total |  |  | 711927.5000 | 995147.0200 | 27630.0000 | 31967.3000 | 0.8148 | 0.8395 | 0.0502 | 0.0720 |

### Difference

| SWE-bench Instance ID | Successes | Success Rate | Input Tokens (Median) | Input Tokens (Mean) | Max Context (Median) | Max Context (Mean) | Cache Hit Rate (Median) | Cache Hit Rate (Mean) | Cost (Median) | Cost (Mean) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| astropy__astropy-7671 | 0.0000 | 0.0000 | 62.1500% | 68.5600% | 99.6400% | 105.2700% | 77.0900% | 77.8600% | 127.9100% | 126.8800% |
| django__django-10554 | 4.0000 | 0.8000 | 60.3800% | 67.0300% | 95.5000% | 100.4700% | 91.5000% | 91.4000% | 138.1000% | 149.7400% |
| matplotlib__matplotlib-26208 | 1.0000 | 0.2000 | 46.5600% | 52.7900% | 88.6200% | 94.1200% | 87.5400% | 88.3700% | 82.0100% | 105.4100% |
| mwaskom__seaborn-3069 | 1.0000 | 0.2000 | 45.2500% | 41.5200% | 76.4000% | 79.2900% | 88.9500% | 87.7400% | 78.5400% | 71.4400% |
| pallets__flask-5014 | 0.0000 | 0.0000 | 75.3600% | 85.7600% | 149.1400% | 139.8300% | 72.1400% | 71.9100% | 192.1200% | 151.6000% |
| pydata__xarray-3993 | -2.0000 | -0.4000 | 43.3200% | 32.4600% | 70.1700% | 66.6200% | 85.4600% | 84.4500% | 58.8100% | 50.2500% |
| pytest-dev__pytest-7324 | 0.0000 | 0.0000 | 10.0500% | 13.4500% | 47.4600% | 53.8200% | 77.0300% | 78.2000% | 28.7100% | 33.5600% |
| scikit-learn__scikit-learn-13496 | 0.0000 | 0.0000 | 193.5900% | 133.1400% | 146.2200% | 128.8000% | 88.2400% | 83.4900% | 161.1200% | 126.4100% |
| sphinx-doc__sphinx-9461 | 0.0000 | 0.0000 | 46.8000% | 46.2600% | 109.0500% | 103.5700% | 89.5200% | 88.6100% | 90.2800% | 82.0800% |
| sympy__sympy-13878 | 4.0000 | 0.8000 | 96.3500% | 78.4500% | 116.9600% | 110.2500% | 93.4600% | 88.3900% | 145.7700% | 114.2100% |
| Total |  |  | 25.1400% | 51.8100% | 68.7700% | 93.1100% | 84.1200% | 85.6900% | 48.0400% | 91.7200% |

## License

Licensed under the [Apache License, Version 2.0](LICENSE).

Copyright 2026 Ryan P. Taylor — ryan@beta-reduce.net

Apache 2.0 grants a patent license and permits commercial and private use,
modification, and redistribution, provided the license and copyright notice are
kept and any modified files carry prominent change notices. It is not a
permissive-without-conditions license: it offers no warranty and no liability.
