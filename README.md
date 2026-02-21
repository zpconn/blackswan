# Crash Prep Optimizer

`crash_prep_optimizer` is a Monte Carlo engine for a specific one-time decision, with an optional C++/CUDA accelerator:

- What fraction of a concentrated stock portfolio, held in post-tax brokerage accounts, should you sell today if you believe a stock market crash will happen in the near future?

It simulates many possible futures (market crashes, recovery paths, layoffs, cash burn, and taxes), evaluates a grid of sell fractions, applies a ruin-probability guardrail, and returns the recommended fraction under the chosen objective.

## Table of Contents

1. [What This Model Solves](#what-this-model-solves)
2. [Core Terms](#core-terms)
3. [How It Works](#how-it-works)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [Configuration](#configuration)
7. [Objective Modes](#objective-modes)
8. [Parallel Execution Model](#parallel-execution-model)
9. [Result Schema](#result-schema)
10. [Troubleshooting](#troubleshooting)
11. [Testing](#testing)
12. [Repository Layout](#repository-layout)
13. [Limitations and Assumptions](#limitations-and-assumptions)

## What This Model Solves

Given beliefs about crash odds/severity and job loss risk, the optimizer answers:

- Sell `x%` of portfolio now, pay tax now, keep the rest invested.
- Then evaluate the long-run wealth and downside risk of that choice across many simulated futures.

The decision is explicitly one-time (`fraction_sold_today`), not a dynamic trading policy.

## Core Terms

- `scenario`: one simulated future path across the full horizon (returns, crash timing/severity, unemployment).
- `strategy option`: one candidate sell fraction from the decision grid.
- `guardrail`: max allowed ruin probability (`risk.max_ruin_probability`).
- `CVaR shortfall`: average tail shortfall (worst `alpha` fraction) relative to `shortfall_floor`.
- `consensus`: minimax-regret compromise between CVaR-style and log-utility objectives.

## How It Works

1. Merge user config onto defaults.
2. Validate config fields and constraints.
3. Generate one shared simulation universe with `n_sims` scenarios.
4. Build decision grid from `min_fraction` to `max_fraction`.
5. For each strategy option (each fraction), run the month-by-month wealth simulation across all scenarios and compute metrics (expected wealth, CVaR shortfall, ruin probability, log utility, etc.).
6. Filter strategies by ruin guardrail.
7. Pick winners for `cvar_shortfall`, `expected_log_utility`, and `consensus`.
8. Return the primary mode result plus cross-checks for the other modes.

## Installation

Python and `numpy` are required. Tests use `pytest`.

```bash
python -m venv .venv
. .venv/bin/activate
pip install numpy pytest
```

### Optional: build the CUDA extension

If you want GPU acceleration through the native extension:

```bash
. .venv/bin/activate
pip install -U pip
pip install scikit-build-core pybind11
pip install -e .
```

This builds `crash_prep_cuda` from:

- `CMakeLists.txt`
- `src/cuda/bindings.cpp`
- `src/cuda/sim_kernel.cu`

### One-command setup/build/benchmark helpers

- `scripts/wsl2_cuda_setup.sh`: installs WSL2 build deps + CUDA toolkit
- `scripts/build_and_benchmark_cuda.sh`: builds extension, runs tests, runs benchmark
- `scripts/benchmark_backends.py`: compares CUDA-extension path vs numpy-streaming path

If CUDA extension import fails, the benchmark now fails fast for the CUDA case instead of silently running a long CPU fallback.

## Quick Start

### Run with defaults

```bash
. .venv/bin/activate
python crash_prep_optimizer.py
```

### Run from Python with custom config

```python
from crash_prep_optimizer import run_monte_carlo_optimization

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

### Safe one-liner shell form

Use single quotes around the whole `-c` script and double quotes inside Python:

```bash
. .venv/bin/activate && python -u -c 'from crash_prep_optimizer import run_monte_carlo_optimization; r = run_monte_carlo_optimization(config={"simulation":{"n_sims":5_000_001,"parallel_enabled":True,"parallel_workers":None},"risk":{"objective_mode":"consensus","max_ruin_probability":0.005}}, verbose=True); print("\nExecution metadata:", r["execution"])'
```

## Configuration

### Merge Behavior

`run_monte_carlo_optimization(config=...)` deep-merges overrides into `DEFAULT_CONFIG`, so you can override only the fields you care about.

### Full Default Config

```python
DEFAULT_CONFIG = {
    "portfolio": {
        "initial_portfolio": 2_500_000.0,
        "initial_basis": 1_250_000.0,
        "ltcg_tax_rate": 0.15,
        "monthly_expenses": 6_000.0,
        "monthly_savings": 8_000.0,
    },
    "market": {
        "volatility_annual": 0.15,
        "cash_yield_annual": 0.04,
        "market_yield_annual": 0.08,
    },
    "simulation": {
        "n_sims": 10_000_000,
        "seed": 42,
        "min_horizon_months": 72,
        "horizon_months": None,
        "max_horizon_months": 240,
        "post_crash_buffer_months": 24,
        "layoff_probabilities_are_annual": True,
        "crash_layoff_min_duration_months": 24,
        "baseline_layoff_min_duration_months": 1,
        "baseline_layoff_max_duration_months": 12,
        "parallel_enabled": False,
        "parallel_workers": None,
        "parallel_start_method": None,
        "parallel_chunk_mode": "fractions",
        "parallel_min_fraction_tasks": 2,
    },
    "decision_grid": {
        "min_fraction": 0.0,
        "max_fraction": 1.0,
        "num_points": 51,
    },
    "risk": {
        "alpha": 0.10,
        "lambda": 0.50,
        "shortfall_floor": None,
        "objective_mode": "cvar_shortfall",
        "max_ruin_probability": 0.005,
        "log_utility_wealth_floor": 1.0,
        "consensus_tolerance": 0.002,
    },
    "beliefs": {
        "prob_crash": 0.80,
        "crash_start_month": {"dist": "uniform", "low": 1, "high": 24},
        "crash_duration_months": {"dist": "triangular", "left": 36, "mode": 60, "right": 96},
        "crash_severity_drop": {"dist": "triangular", "left": 0.45, "mode": 0.50, "right": 0.55},
        "prob_layoff_during_crash": 0.50,
        "prob_layoff_baseline": 0.15,
    },
    "gpu": {
        "enabled": True,
        "device_id": 0,
        "scenario_chunk_size": 262_144,
        "fraction_tile_size": 64,
        "precision_mode": "mixed",
        "cvar_mode": "streaming_near_exact",
        "cvar_refine_pass": True,
        "streams": 2,
        "max_vram_utilization": 0.85,
        "fallback_to_cpu_on_error": False,
        "sample_size": 400_000,
        "prefer_cuda_extension": True,
        "refine_top_k": 5,
        "exact_temp_dir": None,
        "min_chunk_size": 32_768,
        "oom_backoff_factor": 0.5,
    },
}
```

### Distribution Specs

Supported distribution specs for beliefs:

- `beta`: `{"dist":"beta","a":...,"b":...,"low":...,"high":...}`
- `triangular`: `{"dist":"triangular","left":...,"mode":...,"right":...}`
- `uniform`: `{"dist":"uniform","low":...,"high":...}`
- `lognormal`: `{"dist":"lognormal","mean":...,"sigma":...,"clip_low":...,"clip_high":...}`
- scalar number: treated as a fixed constant in all scenarios

### Important Config Notes

- `simulation.horizon_months=None` means horizon is auto-sized to crash timing + post-crash buffer, bounded by `max_horizon_months`.
- `risk.shortfall_floor=None` defaults to `portfolio.initial_portfolio`.
- `decision_grid.num_points=51` with min/max `[0,1]` yields fractions `0%, 2%, ..., 100%`.
- `parallel_chunk_mode` currently supports only `"fractions"`.
- `gpu.enabled=True` activates streaming backend dispatch with safe CPU fallback.
- `gpu.prefer_cuda_extension=True` requires `crash_prep_cuda`; set `False` to use numpy streaming fallback.
- `gpu.scenario_chunk_size` and `gpu.fraction_tile_size` control memory scaling at high `n_sims`.
- `gpu.streams` controls true multi-stream CUDA fraction sharding in the aggregate kernel path.
- `gpu.cvar_mode="streaming_near_exact"` uses bounded-memory tail estimation from sampled scenarios.
- `gpu.cvar_mode="exact_two_pass"` performs an exact second pass using temporary memmap storage.
- `gpu.cvar_refine_pass=True` recomputes exact tail metrics for top candidates in streaming mode.
- `gpu.min_chunk_size` and `gpu.oom_backoff_factor` control automatic CUDA OOM retry behavior.

## Objective Modes

Supported primary modes (`risk.objective_mode`):

- `cvar_shortfall`
- `expected_log_utility`
- `consensus`

### `cvar_shortfall`

- Shortfall per scenario: `max(0, shortfall_floor - final_net_worth)`
- `CVaR(shortfall, alpha)`: mean of worst `alpha` tail
- Objective: `mean_final_net_worth - lambda * cvar_shortfall`
- Higher is better

### `expected_log_utility`

- Utility per scenario: `log(max(final_net_worth, log_utility_wealth_floor))`
- Objective: mean utility
- Higher is better

### `consensus`

- Compute regrets to the best eligible strategy on each objective.
- `cvar_regret`: normalized regret vs best CVaR objective score
- `log_regret`: normalized regret vs best log-utility score
- `max_regret = max(cvar_regret, log_regret)`
- Strategies pass consensus filter if both regrets are `<= consensus_tolerance`
- If pass-set is empty, fallback chooses smallest regret compromise from all eligible strategies

## Parallel Execution Model

Parallelism is across strategy options (fractions), not across scenario count.

- `n_sims` controls total scenario universe size.
- That same universe is reused for each strategy option.
- Worker processes each evaluate one fraction task at a time.
- Execution metadata is returned in `result["execution"]`.

This means:

- not "n_sims per process"
- not "n_sims split across processes"
- yes "fractions split across processes"

### Worker Resolution

- If `parallel_workers=None`, uses `os.cpu_count()` capped by number of tasks.
- If only one worker is effectively available, execution mode is single-thread.
- If parallel execution throws, code falls back to single mode and records `fallback_reason`.

### Start Method

- Default on POSIX prefers `fork` when available.
- Otherwise it uses `spawn` if available.
- You can override via `simulation.parallel_start_method`.

## Result Schema

Top-level return keys from `run_monte_carlo_optimization(...)`:

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

`execution` includes:

- `mode`: `"single"` or `"parallel"`
- `workers_used`
- `backend`
- `start_method`
- `fraction_tasks`
- `fallback_reason`

`objective_comparison` always contains all three objective selections, even if one mode is primary.

## Troubleshooting

### `No strategy satisfies the ruin probability guardrail`

The current assumptions are too strict for `risk.max_ruin_probability`. Options:

- raise `max_ruin_probability`
- reduce downside assumptions
- adjust portfolio/cash-flow assumptions

### Parallel requested but execution shows single mode

Check:

- `simulation.parallel_enabled` is `True`
- `decision_grid` has enough tasks (see `parallel_min_fraction_tasks`)
- runtime allows multiprocessing primitives
- `parallel_workers` is not effectively forced to 1

### GPU requested but backend is not `cuda`

Check:

- `crash_prep_cuda` was built successfully (`pip install -e .`)
- CUDA toolkit/build deps are present in WSL2
- `gpu.prefer_cuda_extension` is set as intended
- `execution.fallback_reason` for precise fallback details

### Exact two-pass mode is slow or disk-heavy

`gpu.cvar_mode="exact_two_pass"` writes temporary memmap files to compute exact tails.  
Use `gpu.exact_temp_dir` to point at a fast disk with enough free space.

### `simulation.horizon_months is shorter than sampled crash windows`

Your fixed horizon is too short for sampled crash timing/duration plus buffer. Increase `horizon_months` or set it back to `None`.

## Testing

Run all tests:

```bash
. .venv/bin/activate
pytest -q
```

Current tests cover:

- config validation for parallel settings
- smoke runs for all objectives in single and parallel modes
- single vs parallel consistency
- consensus fields
- guardrail failure behavior
- worker resolution and single-worker behavior

## Repository Layout

- `crash_prep_optimizer.py`: simulation engine and optimizer
- `gpu_backend.py`: CUDA extension integration and device metadata
- `CMakeLists.txt`: native extension build configuration
- `src/cuda/`: C++/CUDA kernel and pybind bindings
- `tests/test_crash_prep_optimizer.py`: test suite

## Limitations and Assumptions

- This is a scenario model, not a guarantee.
- Output depends heavily on belief distributions and cash-flow assumptions.
- Action space is a one-time sell fraction, not a dynamic multi-step strategy.
- Tax modeling is simplified (single LTCG rate and basis treatment).
- Parallel behavior and speed depend on OS start method and memory constraints.

## Disclaimer

This project is a quantitative decision-support tool, not financial advice.
