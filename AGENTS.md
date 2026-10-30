# Coder Instructions: Histrel

Histrel is a code agent that utilizes a single reasoning state and persistent
tool-call history. It consists of two sub-applications:

- core application (`src/histrel`)
- benchmark harness (`benchmark`) 

The application is built in Python (3.11).

## Vital Documents

Before writing or editing code, you MUST read the Python coding standards and
code style guide at the following locations:

- `agent_resources/code/reference/coding_standards.md`
- `agent_resources/code/reference/style_guide.md`

As you work, fix any easily addressed lack of conformance to the coding
standards and style guide, but ONLY when the fixes are simple and have a
minimal chance of causing regressions.

Before writing or editing tests, you MUST read the best practices guide for
tests at `agent_resources/code/reference/testing/best_practices.md`.
Additionally, you MUST read the specific test standards for the relevant test
type (unit, integration, or property-based), located at:

- `agent_resources/code/reference/testing/unit_tests.md`
- `agent_resources/code/reference/testing/integration_tests.md`
- `agent_resources/code/reference/testing/property_tests.md`

You MUST NEVER use mocks in tests; instead, use fixtures, patching as
necessary. If a fixture is likely to be used over multiple tests, you MUST use
a shared fixture factory or shared fixture whenever possible; if no shared
factory or fixture exists in this situation, create one.

## Repository Hygeine

You MUST ALWAYS adhere to the following upper limits (lower limit for
maintainability index) for Python code in this repository:

```markdown
| Metric | Soft Limit (p90) | Hard Limit (p99) |
|---|---:|---:|
| LOC per module | 500 | 1,000 |
| LOC per function/method | 50 | 100 |
| LOC per class | 250 | 500 |
| Number of methods per class | 12 | 20 |
| Number of classes per module | 5 | 10 |
| Number of functions per module | 15 | 30 |
| Number of submodules per module | 10 | 20 |
| Cyclomatic complexity per function/method | 10 | 20 |
| Cyclomatic complexity per class | 40 | 80 |
| Cyclomatic complexity per module | 50 | 100 |
```

## Existing Tests

Many of the existing tests fail to meet the test standards due to sub-par
performance by the developers who implemented them. As you work, fix any easily
addressed lack of conformance to the test standards, but ONLY when the fixes
are simple and have a minimal chance of causing regressions.

## Development Python and Node runtimes

This project uses the Python venv located at `.venv/`. Ensure it's active
before beginning any work. If it's not, activate it using
`source .venv/bin/activate`.

## Tooling

This project makes use of the following tools (exact versions can be obtained
from `pyproject.toml`):

- Pytest (with pytest-asyncio, pytest-cov, hypothesis, xdoctest,
  pytest-timeout, freezegun): main testing framework and test runner
  - `python -m pytest ...`
  - When running Pytest, ~15% of tests will be skipped unless the `--all` flag
    is included in the `pytest` call
- Ruff: linting and formatting
  - `python -m ruff format DIR`
  - `python -m ruff check DIR`
    - Inspect the outputs of `ruff check` and ONLY run
      `python -m ruff check --fix DIR` when the automatic fixes are safe. When
      some of the automatic fixes are potentially unsafe, you MUST fix the issues
      with potentially unsafe automatic fixes yourself. After confirming all
      such issues have been resolved, use automatic fixes for the remainder.
- Pyright and MyPy: type checking; you MUST ALWAYS use both.
  - `python -m pyright DIR`
  - `python -m mypy DIR`
- deptry: dependency integrity checks
  - `python -m deptry` (from project root)
- interrogate: docstring coverage (ONLY use when specifically instructed to by
  the user)
- Scalene: CPU/GPU/memory profiling (ONLY use when specifically instructed to
  by the user)
- Vulture: identification of potential dead Python code (ONLY use when
  specifically instructed to by the user)
- complexipy: measures cognitive complexity of Python code (ONLY use when
  specifically instructed to by the user)

The following tools are available in the local environment, but are not
Python-managed:

- Tokei: code file counts and LOC-related metrics
  - `tokei DIR` (see `tokei --help` for more options)
- Radon: code complexity metrics
  - `radon {cc,raw,mi,hal} DIR` (see `radon --help` for more options)

Packages are managed using `uv`. However, you MUST NOT attempt to perform
package management yourself; that MUST be performed by the user directly.

## Codebase Details

### Logging

This project uses Loguru for centralized, structured logging. **ALWAYS** import
the Loguru logger with

```python
from loguru import logger
```

When using `logger` **ALWAYS** use it with full structured logging, as follows:

```python
logger.bind(
    component=...,
    event=...
).info(
    "log event text",
    relevant_val=val1,
    other_relevant_val=val2,
    ...
)
```

#### **CRITICAL WARNING**: Loguru Sinks

You MUST NEVER use `logger` from inside a Loguru sink, or any of the
functions/methods called by that sink. This can create feedback loops and race
conditions.

Instead, perform all such introspective logging by writing directly to
`sys.stderr` (be sure to flush the stream after writing).

