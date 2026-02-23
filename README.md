
<img width="1536" height="1024" alt="blackswan-logo" src="https://github.com/user-attachments/assets/ec7a95b6-e398-41e7-bb0e-a5a9d2a6ba36" />

# Blackswan

https://github.com/user-attachments/assets/d154b26e-6413-4ffa-9b55-480b6ef06ef8

Blackswan is a massively parallel GPU-accelerated Monte Carlo decision engine for one specific one-time choice:

- What fraction of a concentrated stock position should you sell today if you believe a crash may happen soon?

It models crash timing/severity, layoffs, cash burn, taxes, and monthly saving behavior across many simulated futures, then recommends the sell-today fraction that best matches your selected objective under a ruin-probability guardrail. On top-flight consumer hardware, a single blackswan run can 100 different sell percentages against hundreds of millions of possible future trajectories in just 5-10m, performing trillions of month-to-month update simulations.

## Table of Contents

1. [What This Actually Does](#what-this-actually-does)
2. [End-to-End Architecture](#end-to-end-architecture)
3. [Algorithm Deep Dive](#algorithm-deep-dive)
4. [Multiprocessing Deep Dive (CPU Path)](#multiprocessing-deep-dive-cpu-path)
5. [GPU Processing Deep Dive (Streaming Path)](#gpu-processing-deep-dive-streaming-path)
6. [Memory Management Deep Dive](#memory-management-deep-dive)
7. [Installation](#installation)
8. [Quick Start](#quick-start)
9. [Blackswan TUI (Charm Stack)](#blackswan-tui-charm-stack)
10. [Configuration Guide](#configuration-guide)
11. [Objective Modes](#objective-modes)
12. [Result Schema](#result-schema)
13. [Troubleshooting](#troubleshooting)
14. [Testing](#testing)
15. [Repository Layout](#repository-layout)
16. [Limitations and Assumptions](#limitations-and-assumptions)

## What This Actually Does

At run time, Blackswan:

1. Merges your config with defaults and validates constraints.
2. Builds a grid of candidate sell fractions (for example, 0%, 2%, ..., 100%).
3. Simulates many scenario paths (`n_sims`) using your crash/layoff beliefs.
4. Evaluates every candidate fraction against the same scenario universe (CPU path) or the same streaming scenario process (GPU/streaming path).
5. Computes risk and utility metrics per fraction.
6. Applies the ruin guardrail (`risk.max_ruin_probability`).
7. Selects winners for all objective modes (`cvar_shortfall`, `expected_log_utility`, `consensus`) and returns the primary-mode recommendation.

Decision scope is intentionally narrow: one immediate sell decision (`fraction_sold_today`), not a dynamic trading policy.

## End-to-End Architecture

```text
+---------------------+      +----------------------+      +-----------------------+
| JSON config +       | ---> | Scenario generation  | ---> | Fraction evaluation   |
| belief distributions|      | (crash + layoff +    |      | (each candidate sell  |
|                     |      | monthly returns)     |      | fraction)             |
+---------------------+      +----------------------+      +-----------------------+
                                                                     |
                                                                     v
                                                        +---------------------------+
                                                        | Metrics per fraction      |
                                                        | mean, p10, CVaR, ruin, log|
                                                        +---------------------------+
                                                                     |
                                                                     v
                                                        +---------------------------+
                                                        | Guardrail filter +         |
                                                        | objective selection        |
                                                        +---------------------------+
                                                                     |
                                                                     v
                                                        +---------------------------+
                                                        | recommended_fraction       |
                                                        +---------------------------+
```

Execution routing inside `run_monte_carlo_optimization`:

```text
run_monte_carlo_optimization
        |
        +-- gpu.enabled = true?
        |       |
        |       +-- yes -> streaming evaluator
        |              |
        |              +-- prefer_cuda_extension and extension available -> CUDA backend
        |              |
        |              +-- otherwise -> numpy_streaming backend
        |              |
        |              +-- on failure + fallback_to_cpu_on_error=true:
        |                     1) retry streaming with numpy backend
        |                     2) if still failing, fallback to full CPU path
        |
        +-- no  -> full CPU universe + per-fraction evaluation
```

## Algorithm Deep Dive

### Core terms

- `scenario`: one full future path across all months.
- `fraction`: candidate sell-today proportion from decision grid.
- `guardrail`: max allowed ruin probability.
- `CVaR shortfall`: mean shortfall in worst `alpha` tail.

### Step 1: Build scenario universe

For each scenario, the model samples:

- crash occurrence probability
- crash start month, duration, severity
- layoff probability in crash and non-crash states

It then generates monthly returns and unemployment flags over the resolved horizon.

Crash/recovery behavior is modeled by applying a down-drift phase then an up-drift phase so the crash severity and duration shape the return path.

### Step 2: Simulate portfolio cashflows for a fixed fraction

Given one sell fraction `f`:

- sell `f * initial_portfolio` at time zero
- pay LTCG tax on sold gains
- hold proceeds in cash and leave the rest invested
- each month:
  - apply market return and cash yield
  - if employed: add monthly savings to stock basis/holdings
  - if unemployed: cover expenses from cash, then liquidate stock if needed (with tax-aware net proceeds)
- at horizon end: apply tax on remaining unrealized gains

The simulator returns per-scenario final net worth and ruin flags.

### Step 3: Score each fraction

For each fraction:

- `expected_final_net_worth = mean(final_net_worth)`
- `p10_final_net_worth = 10th percentile`
- `ruin_probability = mean(ruined)`
- `shortfall = max(0, shortfall_floor - final_net_worth)`
- `cvar_shortfall = mean(worst alpha tail of shortfall)`
- `cvar_objective_score = mean_final - lambda * cvar_shortfall`
- `log_utility_score = mean(log(max(final_net_worth, log_utility_wealth_floor)))`

### Step 4: Guardrail and objective selection

Every objective first filters to strategies with:

- `ruin_probability <= risk.max_ruin_probability`

Then:

- `cvar_shortfall` mode selects highest `cvar_objective_score`.
- `expected_log_utility` mode selects highest `log_utility_score`.
- `consensus` mode minimizes normalized max regret across both objectives, with tolerance-based pass set and fallback to closest compromise if pass set is empty.

## Multiprocessing Deep Dive (CPU Path)

There are two distinct process layers in this project:

1. Run orchestration layer (`python_service/blackswan_service.py`):
   - each submitted run executes in its own `multiprocessing.Process`
2. Optional fraction-evaluation layer (inside `blackswan.py`):
   - if `simulation.parallel_enabled=true`, a `ProcessPoolExecutor` fans out fraction tasks

```text
TUI / caller
    |
    v
+-------------------------------+
| python_service HTTP/SSE layer |
| (threaded server)             |
+---------------+---------------+
                |
                v
      mp.Process per run
                |
                +--> run_monte_carlo_optimization
                         |
                         +--> optional ProcessPoolExecutor
                               (fraction-parallel CPU mode)
```

CPU parallelism is fraction-parallel, not scenario-parallel.

- One universe is generated for the run.
- Tasks are `(index, fraction)` pairs.
- Worker processes evaluate fractions independently.

```text
+---------------------------+
| Main process              |
| - generate universe       |
| - build fraction task list|
+-------------+-------------+
              |
              v
+---------------------------+      +---------------------------+
| Worker process 1          |      | Worker process 2          |
| evaluate fraction i       | ...  | evaluate fraction j       |
+-------------+-------------+      +-------------+-------------+
              \                         /
               \                       /
                +---------------------+
                | collect + order by i |
                +---------------------+
```

Key behavior:

- Enabled by `simulation.parallel_enabled`.
- Worker count defaults to `os.cpu_count()` (capped by task count) if `parallel_workers` is `None`.
- Parallel start method is auto-selected (`fork` preferred on POSIX, else `spawn`) unless overridden.
- If parallel execution fails, engine falls back to single-threaded evaluation and stores `execution.fallback_reason`.

## GPU Processing Deep Dive (Streaming Path)

When `gpu.enabled=true`, the engine uses a streaming evaluator that avoids holding all scenario outputs in RAM at once.

### Streaming loop

```text
for each scenario chunk:
    generate chunk (host-side OR synthetic device generation)
    for each fraction tile:
        run backend kernel/evaluator
        accumulate: sum_final, sum_log, ruin_count
        optionally collect sampled final values for tail estimation
```

### Backends

- `cuda`: uses `blackswan_cuda` extension via `gpu_backend.py`.
- `numpy_streaming`: same chunk/tile flow, but reduced with NumPy on host.

### CVaR modes

- `streaming_near_exact`:
  - keeps bounded samples per fraction (`gpu.sample_size`)
  - computes tail metrics from sampled matrix
  - optional exact refine pass on top candidates (`gpu.cvar_refine_pass`, `gpu.refine_top_k`)
- `exact_two_pass`:
  - performs a full exact tail pass for all fractions using memmap-backed storage
  - requires more host RAM and disk

## Memory Management Deep Dive

Blackswan has explicit memory controls for GPU VRAM, host RAM, and temporary disk.

### 1) VRAM-aware geometry tuning

Initial geometry:

- `chunk_size = min(n_sims, gpu.scenario_chunk_size)`
- `fraction_tile_size = min(n_fractions, gpu.fraction_tile_size)`

If CUDA is used and device memory is known:

- estimate working set bytes
- budget = `max_vram_utilization * (free_or_total_vram)`
- shrink chunk/tile until estimated usage fits budget

```text
estimate(chunk, tile, horizon, sample_count)
        |
        +-- fits budget? yes -> run
        |
        +-- no -> shrink tile or chunk -> re-estimate
```

### 2) Runtime OOM backoff

If CUDA still OOMs at runtime:

- reduce `chunk_size` by `gpu.oom_backoff_factor` until `gpu.min_chunk_size`
- then reduce `fraction_tile_size`
- retry until no further reduction is possible

### 3) Host memory controls for tail reduction

Auto tail-reduction worker count is memory-aware:

- starts from CPU/task limits
- caps workers based on available memory and estimated bytes per worker

For exact/refine passes, resource checks run before execution:

- memmap disk demand must be <= ~80% of free temp-disk capacity
- one-worker host reduction estimate must be <= ~70% of available RAM

When checks fail:

- `exact_two_pass`: run fails with clear message (exact mode is required)
- refine pass: refinement is skipped and reason is recorded

### 4) Lifecycle cleanup

- Large sample matrix is explicitly dropped before exact/refine stages.
- `gc.collect()` is triggered after dropping sampled data.
- Memmap files for exact passes are removed after use.

Exact tail memmap layout (disk-backed):

```text
temp file (float64 memmap)
shape = [subset_fraction_count, n_sims]

row 0 -> final wealth values for fraction A across all scenarios
row 1 -> final wealth values for fraction B across all scenarios
...

write phase: chunk/tile simulation appends columns
reduce phase: percentile/CVaR computed row-by-row
cleanup phase: memmap file deleted
```

## Installation

Python and `numpy` are required. Tests use `pytest`.

```bash
python -m venv .venv
. .venv/bin/activate
pip install numpy pytest
```

### Optional: build CUDA extension

```bash
. .venv/bin/activate
pip install -U pip
pip install scikit-build-core pybind11
pip install -e .
```

Build sources:

- `CMakeLists.txt`
- `src/cuda/bindings.cpp`
- `src/cuda/sim_kernel.cu`

Helper scripts:

- `scripts/wsl2_cuda_setup.sh`
- `scripts/build_and_benchmark_cuda.sh`
- `scripts/benchmark_backends.py`

## Quick Start

### Run with defaults

```bash
. .venv/bin/activate
python blackswan.py
```

### Run from Python

```python
from blackswan import run_monte_carlo_optimization

result = run_monte_carlo_optimization(
    config={
        "simulation": {
            "n_sims": 5_000_001,
            "parallel_enabled": True,
            "parallel_workers": None,
        },
        "risk": {
            "objective_mode": "consensus",
            "max_ruin_probability": 0.005,
        },
    },
    verbose=True,
)

print("Execution metadata:", result["execution"])
print("Recommended fraction:", result["recommended_fraction"])
```

### Shell one-liner

```bash
. .venv/bin/activate && python -u -c 'from blackswan import run_monte_carlo_optimization; r = run_monte_carlo_optimization(config={"simulation":{"n_sims":5000001,"parallel_enabled":True,"parallel_workers":None},"risk":{"objective_mode":"consensus","max_ruin_probability":0.005}}, verbose=True); print("\nExecution metadata:", r["execution"])'
```

## Blackswan TUI (Charm Stack)

`blackswan-tui` is a local terminal cockpit built with Bubble Tea + Bubbles + Lip Gloss.
It launches a local Python service, streams live telemetry, and stores run bundles.

### TUI prerequisites

- Go `1.22+`
- Python virtualenv at `.venv`
- Optional CUDA extension if you want `gpu.prefer_cuda_extension=true`

### Run the TUI

```bash
. .venv/bin/activate
go run ./cmd/blackswan-tui
```

### Core controls

- `ctrl+r`: start run with current JSON config
- `ctrl+x`: cancel active run
- `tab` / `shift+tab`: cycle focus panes
- `up` / `down` or mouse wheel: scroll focused pane
- `enter`: load selected history bundle
- `ctrl+l`: clear telemetry
- `q` or `ctrl+c`: quit

### Service API used by the TUI

- `POST /runs`
- `GET /runs/{id}`
- `GET /runs/{id}/stream` (SSE)
- `POST /runs/{id}/cancel`

## Configuration Guide

### Merge behavior

`run_monte_carlo_optimization(config=...)` deep-merges your overrides onto `DEFAULT_CONFIG`.

### Most important knobs

- Simulation scale:
  - `simulation.n_sims`
  - `decision_grid.num_points`
- Risk behavior:
  - `risk.objective_mode`
  - `risk.max_ruin_probability`
  - `risk.alpha`, `risk.lambda`, `risk.shortfall_floor`
- CPU parallel:
  - `simulation.parallel_enabled`
  - `simulation.parallel_workers`
  - `simulation.parallel_start_method`
- GPU/streaming:
  - `gpu.enabled`
  - `gpu.prefer_cuda_extension`
  - `gpu.cvar_mode`
  - `gpu.scenario_chunk_size`, `gpu.fraction_tile_size`
  - `gpu.max_vram_utilization`
  - `gpu.fallback_to_cpu_on_error`

### Distribution specs for beliefs

- `beta`: `{"dist":"beta","a":...,"b":...,"low":...,"high":...}`
- `triangular`: `{"dist":"triangular","left":...,"mode":...,"right":...}`
- `uniform`: `{"dist":"uniform","low":...,"high":...}`
- `lognormal`: `{"dist":"lognormal","mean":...,"sigma":...,"clip_low":...,"clip_high":...}`
- scalar number: fixed value for all scenarios

### Inspect full defaults

The authoritative defaults live in `blackswan.py` as `DEFAULT_CONFIG`.

```bash
python - <<'PY'
from pprint import pprint
from blackswan import DEFAULT_CONFIG
pprint(DEFAULT_CONFIG)
PY
```

## Objective Modes

Supported primary modes (`risk.objective_mode`):

- `cvar_shortfall`
- `expected_log_utility`
- `consensus`

### cvar_shortfall

- Shortfall per scenario: `max(0, shortfall_floor - final_net_worth)`
- Objective: `mean_final_net_worth - lambda * CVaR_alpha(shortfall)`
- Higher is better

### expected_log_utility

- Utility per scenario: `log(max(final_net_worth, log_utility_wealth_floor))`
- Objective: mean utility
- Higher is better

### consensus

- Compute normalized regrets against best eligible CVaR and log-utility strategies
- Minimize worst-case regret (`max_regret`)
- If no candidate is within `consensus_tolerance` on both regrets, choose nearest compromise

## Result Schema

Top-level result keys:

- `recommended_fraction`
- `objective_score`
- `expected_final_net_worth`
- `cvar_shortfall`
- `ruin_probability`
- `score_gap_vs_second_best`
- `local_score_margin`
- `top_candidates`
- `primary_objective_mode`
- `max_ruin_probability`
- `objective_comparison`
- `diagnostics_by_fraction`
- `universe_summary`
- `execution`

Execution metadata includes:

- universal fields: `mode`, `workers_used`, `backend`, `start_method`, `fraction_tasks`, `fallback_reason`
- streaming/CUDA fields when applicable: `gpu_name`, `driver_version`, `device_memory_total_mb`, `device_memory_free_mb`, `chunk_size_used`, `fraction_tile_used`, `precision_mode`, `cvar_mode`, `kernel_time_ms`, `transfer_time_ms`, `reduction_time_ms`

## Troubleshooting

### `No strategy satisfies the ruin probability guardrail`

Assumptions are too strict for current guardrail. You can:

- raise `risk.max_ruin_probability`
- reduce downside assumptions
- adjust portfolio/cashflow assumptions

### Parallel requested but execution shows single

Check:

- `simulation.parallel_enabled` is `True`
- `decision_grid.num_points` is high enough vs `parallel_min_fraction_tasks`
- `parallel_workers` is not effectively `1`
- start method compatibility on your platform

### GPU requested but backend is not `cuda`

Check:

- CUDA extension build (`pip install -e .`)
- `gpu.prefer_cuda_extension`
- `execution.fallback_reason`
- `gpu.fallback_to_cpu_on_error`

### `exact_two_pass` is slow or disk-heavy

`gpu.cvar_mode="exact_two_pass"` writes a large memmap file for exact tails.
Set `gpu.exact_temp_dir` to a fast disk with plenty of free space.

## Testing

Run tests:

```bash
. .venv/bin/activate
pytest -q
```

Coverage includes:

- config validation
- objective smoke tests
- single vs parallel consistency
- consensus outputs
- guardrail failures
- worker resolution and GPU fallback behavior

## Repository Layout

- `blackswan.py`: simulation engine + objective selection
- `gpu_backend.py`: CUDA extension adapter + GPU metadata
- `src/cuda/`: CUDA kernels + bindings
- `python_service/blackswan_service.py`: local authenticated HTTP/SSE run service
- `cmd/blackswan-tui/main.go`: TUI entrypoint
- `internal/app/`: Bubble Tea state/view logic
- `internal/service/`: Go service manager and SSE client
- `internal/storage/`: run bundle persistence
- `tests/test_blackswan.py`: test suite

## Limitations and Assumptions

- Scenario model, not a guarantee.
- Output is highly sensitive to beliefs and cashflow assumptions.
- Action space is a one-time sell fraction, not a dynamic strategy.
- Tax model is intentionally simplified.
- Runtime and memory behavior depend on hardware, start method, and config.

## Disclaimer

This project is a quantitative decision-support tool, not financial advice.
