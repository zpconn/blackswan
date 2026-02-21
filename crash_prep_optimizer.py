import concurrent.futures
from copy import deepcopy
import multiprocessing
import os
import tempfile
import time

import numpy as np

import gpu_backend


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
        # When layoff_probabilities_are_annual is True, these are annual probabilities.
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

SUPPORTED_DISTRIBUTIONS = {"beta", "triangular", "uniform", "lognormal"}
SUPPORTED_OBJECTIVES = {"cvar_shortfall", "expected_log_utility", "consensus"}
SUPPORTED_PARALLEL_CHUNK_MODES = {"fractions"}
SUPPORTED_GPU_PRECISION_MODES = {"mixed", "fp64_strict", "fp32_fast"}
SUPPORTED_GPU_CVAR_MODES = {"streaming_near_exact", "exact_two_pass"}

_PARALLEL_UNIVERSE = None
_PARALLEL_CONFIG = None


def _deep_merge(base, overrides):
    merged = deepcopy(base)
    _deep_merge_in_place(merged, overrides)
    return merged


def _deep_merge_in_place(target, overrides):
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge_in_place(target[key], value)
        else:
            target[key] = value


def _validate_distribution_spec(name, spec):
    if isinstance(spec, (int, float)):
        return

    if not isinstance(spec, dict):
        raise ValueError(f"{name} must be a number or a distribution dict.")

    dist = spec.get("dist")
    if dist not in SUPPORTED_DISTRIBUTIONS:
        allowed = ", ".join(sorted(SUPPORTED_DISTRIBUTIONS))
        raise ValueError(f"{name}.dist must be one of: {allowed}")

    if dist == "beta":
        for field in ("a", "b", "low", "high"):
            if field not in spec:
                raise ValueError(f"{name} beta distribution missing '{field}'.")
        if spec["a"] <= 0 or spec["b"] <= 0:
            raise ValueError(f"{name} beta requires a>0 and b>0.")
        if spec["low"] >= spec["high"]:
            raise ValueError(f"{name} beta requires low < high.")

    if dist == "triangular":
        for field in ("left", "mode", "right"):
            if field not in spec:
                raise ValueError(f"{name} triangular distribution missing '{field}'.")
        left = spec["left"]
        mode = spec["mode"]
        right = spec["right"]
        if not (left <= mode <= right):
            raise ValueError(f"{name} triangular requires left <= mode <= right.")

    if dist == "uniform":
        for field in ("low", "high"):
            if field not in spec:
                raise ValueError(f"{name} uniform distribution missing '{field}'.")
        if spec["low"] > spec["high"]:
            raise ValueError(f"{name} uniform requires low <= high.")

    if dist == "lognormal":
        for field in ("mean", "sigma"):
            if field not in spec:
                raise ValueError(f"{name} lognormal distribution missing '{field}'.")
        if spec["sigma"] < 0:
            raise ValueError(f"{name} lognormal requires sigma >= 0.")
        if "clip_low" in spec and "clip_high" in spec and spec["clip_low"] > spec["clip_high"]:
            raise ValueError(f"{name} lognormal requires clip_low <= clip_high.")


def validate_config(config):
    for key in ("portfolio", "market", "simulation", "decision_grid", "risk", "beliefs", "gpu"):
        if key not in config:
            raise ValueError(f"Missing top-level config section '{key}'.")

    portfolio = config["portfolio"]
    market = config["market"]
    simulation = config["simulation"]
    decision_grid = config["decision_grid"]
    risk = config["risk"]
    beliefs = config["beliefs"]
    gpu = config["gpu"]

    if portfolio["initial_portfolio"] <= 0:
        raise ValueError("portfolio.initial_portfolio must be > 0.")
    if portfolio["initial_basis"] < 0:
        raise ValueError("portfolio.initial_basis must be >= 0.")
    if not (0 <= portfolio["ltcg_tax_rate"] < 1):
        raise ValueError("portfolio.ltcg_tax_rate must be in [0, 1).")
    if portfolio["monthly_expenses"] < 0 or portfolio["monthly_savings"] < 0:
        raise ValueError("portfolio.monthly_expenses and monthly_savings must be >= 0.")

    if market["volatility_annual"] < 0:
        raise ValueError("market.volatility_annual must be >= 0.")
    if market["cash_yield_annual"] < -0.95 or market["market_yield_annual"] < -0.95:
        raise ValueError("market yields are unrealistically low.")

    if simulation["n_sims"] <= 0:
        raise ValueError("simulation.n_sims must be > 0.")
    if simulation["min_horizon_months"] <= 0:
        raise ValueError("simulation.min_horizon_months must be > 0.")
    if simulation["max_horizon_months"] <= 0:
        raise ValueError("simulation.max_horizon_months must be > 0.")
    if simulation["crash_layoff_min_duration_months"] <= 0:
        raise ValueError("simulation.crash_layoff_min_duration_months must be > 0.")
    if simulation["baseline_layoff_min_duration_months"] <= 0:
        raise ValueError("simulation.baseline_layoff_min_duration_months must be > 0.")
    if simulation["baseline_layoff_max_duration_months"] <= 0:
        raise ValueError("simulation.baseline_layoff_max_duration_months must be > 0.")
    if (
        simulation["baseline_layoff_min_duration_months"]
        > simulation["baseline_layoff_max_duration_months"]
    ):
        raise ValueError(
            "simulation.baseline_layoff_min_duration_months cannot exceed "
            "simulation.baseline_layoff_max_duration_months."
        )
    if simulation["horizon_months"] is not None and simulation["horizon_months"] <= 0:
        raise ValueError("simulation.horizon_months must be positive or None.")
    if not isinstance(simulation["layoff_probabilities_are_annual"], bool):
        raise ValueError("simulation.layoff_probabilities_are_annual must be a bool.")
    if not isinstance(simulation["parallel_enabled"], bool):
        raise ValueError("simulation.parallel_enabled must be a bool.")
    if simulation["parallel_workers"] is not None and simulation["parallel_workers"] <= 0:
        raise ValueError("simulation.parallel_workers must be > 0 or None.")
    if simulation["parallel_start_method"] is not None:
        available_methods = multiprocessing.get_all_start_methods()
        if simulation["parallel_start_method"] not in available_methods:
            available = ", ".join(available_methods)
            raise ValueError(
                "simulation.parallel_start_method must be one of "
                f"[{available}] or None."
            )
    if simulation["parallel_chunk_mode"] not in SUPPORTED_PARALLEL_CHUNK_MODES:
        allowed = ", ".join(sorted(SUPPORTED_PARALLEL_CHUNK_MODES))
        raise ValueError(f"simulation.parallel_chunk_mode must be one of: {allowed}")
    if simulation["parallel_min_fraction_tasks"] <= 0:
        raise ValueError("simulation.parallel_min_fraction_tasks must be > 0.")

    if not (0.0 <= decision_grid["min_fraction"] <= decision_grid["max_fraction"] <= 1.0):
        raise ValueError("decision_grid fractions must satisfy 0 <= min <= max <= 1.")
    if decision_grid["num_points"] < 2:
        raise ValueError("decision_grid.num_points must be at least 2.")

    if not (0 < risk["alpha"] <= 1):
        raise ValueError("risk.alpha must be in (0, 1].")
    if risk["lambda"] < 0:
        raise ValueError("risk.lambda must be >= 0.")
    if risk["shortfall_floor"] is not None and risk["shortfall_floor"] < 0:
        raise ValueError("risk.shortfall_floor must be >= 0 or None.")
    if risk["objective_mode"] not in SUPPORTED_OBJECTIVES:
        allowed = ", ".join(sorted(SUPPORTED_OBJECTIVES))
        raise ValueError(f"risk.objective_mode must be one of: {allowed}")
    if not (0.0 <= risk["max_ruin_probability"] <= 1.0):
        raise ValueError("risk.max_ruin_probability must be in [0, 1].")
    if risk["log_utility_wealth_floor"] <= 0:
        raise ValueError("risk.log_utility_wealth_floor must be > 0.")
    if not (0.0 <= risk["consensus_tolerance"] <= 1.0):
        raise ValueError("risk.consensus_tolerance must be in [0, 1].")

    if not isinstance(gpu["enabled"], bool):
        raise ValueError("gpu.enabled must be a bool.")
    if gpu["device_id"] < 0:
        raise ValueError("gpu.device_id must be >= 0.")
    if gpu["scenario_chunk_size"] <= 0:
        raise ValueError("gpu.scenario_chunk_size must be > 0.")
    if gpu["fraction_tile_size"] <= 0:
        raise ValueError("gpu.fraction_tile_size must be > 0.")
    if gpu["precision_mode"] not in SUPPORTED_GPU_PRECISION_MODES:
        allowed = ", ".join(sorted(SUPPORTED_GPU_PRECISION_MODES))
        raise ValueError(f"gpu.precision_mode must be one of: {allowed}")
    if gpu["cvar_mode"] not in SUPPORTED_GPU_CVAR_MODES:
        allowed = ", ".join(sorted(SUPPORTED_GPU_CVAR_MODES))
        raise ValueError(f"gpu.cvar_mode must be one of: {allowed}")
    if not isinstance(gpu["cvar_refine_pass"], bool):
        raise ValueError("gpu.cvar_refine_pass must be a bool.")
    if gpu["streams"] <= 0:
        raise ValueError("gpu.streams must be > 0.")
    if not (0.1 <= gpu["max_vram_utilization"] <= 0.98):
        raise ValueError("gpu.max_vram_utilization must be in [0.1, 0.98].")
    if not isinstance(gpu["fallback_to_cpu_on_error"], bool):
        raise ValueError("gpu.fallback_to_cpu_on_error must be a bool.")
    if gpu["sample_size"] <= 0:
        raise ValueError("gpu.sample_size must be > 0.")
    if not isinstance(gpu["prefer_cuda_extension"], bool):
        raise ValueError("gpu.prefer_cuda_extension must be a bool.")
    if gpu["refine_top_k"] <= 0:
        raise ValueError("gpu.refine_top_k must be > 0.")
    if gpu["exact_temp_dir"] is not None and not isinstance(gpu["exact_temp_dir"], str):
        raise ValueError("gpu.exact_temp_dir must be a string path or None.")
    if gpu["min_chunk_size"] <= 0:
        raise ValueError("gpu.min_chunk_size must be > 0.")
    if not (0.1 <= gpu["oom_backoff_factor"] < 1.0):
        raise ValueError("gpu.oom_backoff_factor must be in [0.1, 1.0).")

    required_belief_keys = (
        "prob_crash",
        "crash_start_month",
        "crash_duration_months",
        "crash_severity_drop",
        "prob_layoff_during_crash",
        "prob_layoff_baseline",
    )
    for belief_key in required_belief_keys:
        if belief_key not in beliefs:
            raise ValueError(f"Missing beliefs.{belief_key}")
        _validate_distribution_spec(f"beliefs.{belief_key}", beliefs[belief_key])


def _sample_distribution(spec, size, rng):
    if isinstance(spec, (int, float)):
        return np.full(size, float(spec), dtype=np.float64)

    dist = spec["dist"]
    if dist == "beta":
        raw = rng.beta(spec["a"], spec["b"], size=size)
        return spec["low"] + (spec["high"] - spec["low"]) * raw
    if dist == "triangular":
        return rng.triangular(spec["left"], spec["mode"], spec["right"], size=size)
    if dist == "uniform":
        return rng.uniform(spec["low"], spec["high"], size=size)
    if dist == "lognormal":
        values = rng.lognormal(mean=spec["mean"], sigma=spec["sigma"], size=size)
        if "clip_low" in spec:
            values = np.maximum(values, spec["clip_low"])
        if "clip_high" in spec:
            values = np.minimum(values, spec["clip_high"])
        return values

    raise ValueError(f"Unsupported distribution: {dist}")


def _sample_probability(spec, size, rng):
    values = _sample_distribution(spec, size, rng)
    return np.clip(values, 0.0, 1.0)


def _sample_positive_int(spec, size, rng, minimum=1):
    sampled = _sample_distribution(spec, size, rng)
    if not np.all(np.isfinite(sampled)):
        raise ValueError("Integer distribution sampling produced non-finite values.")
    values = np.rint(sampled).astype(np.int32)
    return np.maximum(values, minimum)


def _compute_cvar(losses, alpha):
    if losses.size == 0:
        return 0.0
    tail_count = max(1, int(np.ceil(alpha * losses.size)))
    tail = np.partition(losses, losses.size - tail_count)[-tail_count:]
    return float(np.mean(tail))


def _generate_universe(config):
    beliefs = config["beliefs"]
    market = config["market"]
    simulation = config["simulation"]

    n_sims = simulation["n_sims"]
    rng = np.random.default_rng(simulation["seed"])

    prob_crash = _sample_probability(beliefs["prob_crash"], n_sims, rng)
    crash_start = _sample_positive_int(beliefs["crash_start_month"], n_sims, rng, minimum=1)
    crash_duration = _sample_positive_int(beliefs["crash_duration_months"], n_sims, rng, minimum=1)
    crash_severity = np.clip(
        _sample_distribution(beliefs["crash_severity_drop"], n_sims, rng),
        1e-4,
        0.95,
    )
    prob_layoff_during_crash = _sample_probability(beliefs["prob_layoff_during_crash"], n_sims, rng)
    prob_layoff_baseline = _sample_probability(beliefs["prob_layoff_baseline"], n_sims, rng)

    is_crash = rng.random(n_sims) < prob_crash

    crash_end = crash_start + crash_duration
    down_months = np.maximum(1, crash_duration // 2)
    up_months = np.maximum(1, crash_duration - down_months)
    down_end = crash_start + down_months

    min_horizon = simulation["min_horizon_months"]
    post_crash_buffer = simulation["post_crash_buffer_months"]
    max_horizon = simulation["max_horizon_months"]
    configured_horizon = simulation["horizon_months"]

    if np.any(is_crash):
        required_horizon = int(np.max(crash_end[is_crash])) + post_crash_buffer
    else:
        required_horizon = min_horizon
    required_horizon = max(required_horizon, min_horizon)

    if configured_horizon is not None:
        if configured_horizon < required_horizon:
            raise ValueError(
                "simulation.horizon_months is shorter than sampled crash windows "
                f"(required at least {required_horizon}, got {configured_horizon})."
            )
        horizon_months = configured_horizon
    else:
        horizon_months = required_horizon

    if horizon_months > max_horizon:
        raise ValueError(
            "Sampled crash windows require a horizon larger than "
            f"simulation.max_horizon_months ({horizon_months} > {max_horizon}). "
            "Adjust beliefs or increase max_horizon_months."
        )

    sigma = market["volatility_annual"] / np.sqrt(12.0)
    monthly_market_drift = market["market_yield_annual"] / 12.0

    monthly_returns = rng.normal(0.0, sigma, size=(n_sims, horizon_months)).astype(np.float32)

    if np.any(is_crash):
        mu_down = (1.0 - crash_severity) ** (1.0 / down_months) - 1.0
        mu_up = (1.0 / (1.0 - crash_severity)) ** (1.0 / up_months) - 1.0

        for month in range(1, horizon_months + 1):
            drift = np.full(n_sims, monthly_market_drift, dtype=np.float32)

            mask_down = is_crash & (month >= crash_start) & (month < down_end)
            if np.any(mask_down):
                drift[mask_down] = mu_down[mask_down].astype(np.float32)

            mask_up = is_crash & (month >= down_end) & (month < crash_end)
            if np.any(mask_up):
                drift[mask_up] = mu_up[mask_up].astype(np.float32)

            monthly_returns[:, month - 1] += drift
    else:
        monthly_returns += np.float32(monthly_market_drift)

    # Keep market index positive in extreme tails.
    np.clip(monthly_returns, -0.95, 1.5, out=monthly_returns)

    layoff_probabilities_are_annual = simulation["layoff_probabilities_are_annual"]
    crash_layoff_min_duration = simulation["crash_layoff_min_duration_months"]
    baseline_layoff_min_duration = simulation["baseline_layoff_min_duration_months"]
    baseline_layoff_max_duration = simulation["baseline_layoff_max_duration_months"]

    layoff_start = np.full(n_sims, horizon_months + 1, dtype=np.int32)
    layoff_end = np.full(n_sims, horizon_months + 1, dtype=np.int32)
    will_lose_job = np.zeros(n_sims, dtype=bool)

    crash_idx = np.flatnonzero(is_crash)
    if crash_idx.size > 0:
        crash_layoff_event_prob = prob_layoff_during_crash[crash_idx]
        if layoff_probabilities_are_annual:
            crash_years = (crash_end[crash_idx] - crash_start[crash_idx]) / 12.0
            crash_layoff_event_prob = 1.0 - np.power(1.0 - crash_layoff_event_prob, crash_years)
            crash_layoff_event_prob = np.clip(crash_layoff_event_prob, 0.0, 1.0)

        crash_layoff = rng.random(crash_idx.size) < crash_layoff_event_prob
        crash_layoff_idx = crash_idx[crash_layoff]
        if crash_layoff_idx.size > 0:
            latest_start = crash_end[crash_layoff_idx] - crash_layoff_min_duration
            eligible = latest_start >= crash_start[crash_layoff_idx]
            eligible_idx = crash_layoff_idx[eligible]

            if eligible_idx.size > 0:
                eligible_latest_start = latest_start[eligible]
                start_window = eligible_latest_start - crash_start[eligible_idx] + 1
                offsets = np.floor(rng.random(eligible_idx.size) * start_window).astype(np.int32)
                starts = crash_start[eligible_idx] + offsets

                max_duration = crash_end[eligible_idx] - starts
                duration_span = max_duration - crash_layoff_min_duration + 1
                duration_offsets = np.floor(rng.random(eligible_idx.size) * duration_span).astype(
                    np.int32
                )
                durations = crash_layoff_min_duration + duration_offsets

                will_lose_job[eligible_idx] = True
                layoff_start[eligible_idx] = starts
                layoff_end[eligible_idx] = starts + durations

    non_crash_idx = np.flatnonzero(~is_crash)
    if non_crash_idx.size > 0:
        baseline_layoff_event_prob = prob_layoff_baseline[non_crash_idx]
        if layoff_probabilities_are_annual:
            horizon_years = horizon_months / 12.0
            baseline_layoff_event_prob = 1.0 - np.power(
                1.0 - baseline_layoff_event_prob,
                horizon_years,
            )
            baseline_layoff_event_prob = np.clip(baseline_layoff_event_prob, 0.0, 1.0)

        baseline_layoff = rng.random(non_crash_idx.size) < baseline_layoff_event_prob
        baseline_layoff_idx = non_crash_idx[baseline_layoff]
        if baseline_layoff_idx.size > 0:
            latest_start = horizon_months - baseline_layoff_min_duration + 1
            if latest_start >= 1:
                starts = rng.integers(1, latest_start + 1, size=baseline_layoff_idx.size, dtype=np.int32)
                max_duration = np.minimum(
                    baseline_layoff_max_duration,
                    horizon_months - starts + 1,
                )
                duration_span = max_duration - baseline_layoff_min_duration + 1
                duration_offsets = np.floor(
                    rng.random(baseline_layoff_idx.size) * duration_span
                ).astype(np.int32)
                durations = baseline_layoff_min_duration + duration_offsets

                will_lose_job[baseline_layoff_idx] = True
                layoff_start[baseline_layoff_idx] = starts
                layoff_end[baseline_layoff_idx] = starts + durations

    months = np.arange(1, horizon_months + 1, dtype=np.int32)[None, :]
    unemployment_matrix = will_lose_job[:, None] & (months >= layoff_start[:, None]) & (
        months < layoff_end[:, None]
    )

    summary = {
        "horizon_months": int(horizon_months),
        "realized_crash_rate": float(np.mean(is_crash)),
        "mean_crash_start_if_crash": float(np.mean(crash_start[is_crash])) if np.any(is_crash) else None,
        "mean_crash_duration_if_crash": float(np.mean(crash_duration[is_crash])) if np.any(is_crash) else None,
        "mean_crash_severity_if_crash": float(np.mean(crash_severity[is_crash])) if np.any(is_crash) else None,
        "layoff_rate_if_crash": float(np.mean(will_lose_job[is_crash])) if np.any(is_crash) else 0.0,
        "layoff_rate_if_no_crash": float(np.mean(will_lose_job[~is_crash]))
        if np.any(~is_crash)
        else 0.0,
    }

    return {
        "n_sims": n_sims,
        "horizon_months": horizon_months,
        "monthly_returns": monthly_returns,
        "unemployment_matrix": unemployment_matrix,
        "summary": summary,
    }


def _simulate_strategy_fraction(fraction_sold, universe, config):
    portfolio = config["portfolio"]
    market = config["market"]
    n_sims = universe["n_sims"]
    horizon = universe["horizon_months"]

    initial_portfolio = portfolio["initial_portfolio"]
    initial_basis = portfolio["initial_basis"]
    ltcg_tax_rate = portfolio["ltcg_tax_rate"]
    monthly_expenses = portfolio["monthly_expenses"]
    monthly_savings = portfolio["monthly_savings"]
    r_cash = market["cash_yield_annual"] / 12.0

    gross_sold = fraction_sold * initial_portfolio
    basis_sold = fraction_sold * initial_basis
    capital_gains = max(0.0, gross_sold - basis_sold)
    tax_paid = capital_gains * ltcg_tax_rate

    cash = np.full(n_sims, gross_sold - tax_paid, dtype=np.float64)
    stocks = np.full(n_sims, (1.0 - fraction_sold) * initial_portfolio, dtype=np.float64)
    basis = np.full(n_sims, (1.0 - fraction_sold) * initial_basis, dtype=np.float64)
    market_index = np.ones(n_sims, dtype=np.float64)
    ruined = np.zeros(n_sims, dtype=bool)

    monthly_returns = universe["monthly_returns"]
    unemployment_matrix = universe["unemployment_matrix"]

    for month_idx in range(horizon):
        market_index *= 1.0 + monthly_returns[:, month_idx]
        cash *= 1.0 + r_cash

        is_unemployed = unemployment_matrix[:, month_idx]
        is_employed = ~is_unemployed

        if np.any(is_employed):
            stocks[is_employed] += monthly_savings / market_index[is_employed]
            basis[is_employed] += monthly_savings

        need = np.where(is_unemployed, monthly_expenses, 0.0)

        from_cash = np.minimum(need, cash)
        cash -= from_cash
        need -= from_cash

        need_more = need > 1e-9
        if np.any(need_more):
            no_assets = need_more & (stocks <= 1e-12)
            ruined[no_assets] = True

            active = need_more & (stocks > 1e-12)
            if np.any(active):
                active_idx = np.where(active)[0]

                s_act = stocks[active_idx]
                b_act = basis[active_idx]
                m_act = market_index[active_idx]
                need_act = need[active_idx]

                basis_per_unit = b_act / s_act
                gain_per_unit = np.maximum(0.0, m_act - basis_per_unit)
                tax_per_unit = gain_per_unit * ltcg_tax_rate
                net_per_unit = np.maximum(1e-12, m_act - tax_per_unit)

                units_to_sell = need_act / net_per_unit
                overdraw = units_to_sell > s_act
                units_to_sell[overdraw] = s_act[overdraw]

                proceeds = units_to_sell * net_per_unit
                remaining_need = need_act - proceeds
                ruined[active_idx] |= remaining_need > 1e-9

                s_new = s_act - units_to_sell
                b_new = b_act * (s_new / s_act)

                stocks[active_idx] = s_new
                basis[active_idx] = b_new

    portfolio_value = stocks * market_index
    final_gains = np.maximum(0.0, portfolio_value - basis)
    final_tax = final_gains * ltcg_tax_rate
    final_net_worth = cash + portfolio_value - final_tax
    ruined |= final_net_worth <= 0

    return final_net_worth, ruined


def _objective_key_for_mode(objective_mode):
    if objective_mode == "cvar_shortfall":
        return "cvar_objective_score"
    if objective_mode == "expected_log_utility":
        return "log_utility_score"
    if objective_mode == "consensus":
        return "consensus_score"
    raise ValueError(f"Unsupported objective mode: {objective_mode}")


def _objective_label(objective_mode):
    if objective_mode == "cvar_shortfall":
        return "Mean - lambda*CVaR"
    if objective_mode == "expected_log_utility":
        return "E[log(wealth)]"
    if objective_mode == "consensus":
        return "Consensus score"
    raise ValueError(f"Unsupported objective mode: {objective_mode}")


def _format_objective_for_table(value, objective_mode):
    if objective_mode == "cvar_shortfall":
        return f"${value:15,.0f}"
    if objective_mode == "expected_log_utility":
        return f"{value:16.6f}"
    if objective_mode == "consensus":
        return f"{value:16.6%}"
    raise ValueError(f"Unsupported objective mode: {objective_mode}")


def _format_objective_for_summary(value, objective_mode):
    if objective_mode == "cvar_shortfall":
        return f"${value:,.0f}"
    if objective_mode == "expected_log_utility":
        return f"{value:.6f}"
    if objective_mode == "consensus":
        return f"{(-value):.6%} max regret (lower is better)"
    raise ValueError(f"Unsupported objective mode: {objective_mode}")


def _score_strategy(final_net_worth, ruined, config):
    risk = config["risk"]
    portfolio = config["portfolio"]

    mean_final_net_worth = float(np.mean(final_net_worth))
    p10_final_net_worth = float(np.percentile(final_net_worth, 10))
    ruin_probability = float(np.mean(ruined))

    shortfall_floor = risk["shortfall_floor"]
    if shortfall_floor is None:
        shortfall_floor = portfolio["initial_portfolio"]

    shortfall = np.maximum(0.0, shortfall_floor - final_net_worth)
    cvar_shortfall = _compute_cvar(shortfall, risk["alpha"])
    cvar_objective_score = mean_final_net_worth - risk["lambda"] * cvar_shortfall

    floored_wealth = np.maximum(final_net_worth, risk["log_utility_wealth_floor"])
    log_utility_score = float(np.mean(np.log(floored_wealth)))

    objective_mode = risk["objective_mode"]
    if objective_mode == "cvar_shortfall":
        display_score = cvar_objective_score
    elif objective_mode == "expected_log_utility":
        display_score = log_utility_score
    else:
        # Consensus selection is computed after evaluating all fractions.
        display_score = 0.0

    return {
        "expected_final_net_worth": mean_final_net_worth,
        "p10_final_net_worth": p10_final_net_worth,
        "cvar_shortfall": cvar_shortfall,
        "ruin_probability": ruin_probability,
        "cvar_objective_score": cvar_objective_score,
        "log_utility_score": log_utility_score,
        "score": display_score,
    }


def _iter_chunk_ranges(total, chunk_size):
    start = 0
    while start < total:
        end = min(total, start + chunk_size)
        yield start, end
        start = end


def _build_sampling_indices(total, sample_size, seed):
    if sample_size >= total:
        return np.arange(total, dtype=np.int64)

    rng = np.random.default_rng(seed)
    selected = set()
    while len(selected) < sample_size:
        remaining = sample_size - len(selected)
        draws = rng.integers(0, total, size=remaining * 2, dtype=np.int64)
        selected.update(int(x) for x in draws)
    sampled = np.fromiter(selected, dtype=np.int64, count=len(selected))
    if sampled.size > sample_size:
        keep = rng.choice(sampled.size, size=sample_size, replace=False)
        sampled = sampled[keep]
    sampled.sort()
    return sampled


def _pre_scan_required_horizon(config, chunk_size):
    beliefs = config["beliefs"]
    simulation = config["simulation"]
    n_sims = simulation["n_sims"]

    rng = np.random.default_rng(simulation["seed"])
    max_crash_end = simulation["min_horizon_months"]
    for start, end in _iter_chunk_ranges(n_sims, chunk_size):
        size = end - start
        prob_crash = _sample_probability(beliefs["prob_crash"], size, rng)
        crash_start = _sample_positive_int(beliefs["crash_start_month"], size, rng, minimum=1)
        crash_duration = _sample_positive_int(beliefs["crash_duration_months"], size, rng, minimum=1)
        is_crash = rng.random(size) < prob_crash
        if np.any(is_crash):
            local_max = int(np.max(crash_start[is_crash] + crash_duration[is_crash]))
            if local_max > max_crash_end:
                max_crash_end = local_max

    return max(max_crash_end + simulation["post_crash_buffer_months"], simulation["min_horizon_months"])


def _resolve_streaming_horizon(config):
    simulation = config["simulation"]
    chunk_size = min(simulation["n_sims"], config["gpu"]["scenario_chunk_size"])
    required_horizon = _pre_scan_required_horizon(config, chunk_size)
    configured_horizon = simulation["horizon_months"]
    max_horizon = simulation["max_horizon_months"]

    if configured_horizon is not None:
        if configured_horizon < required_horizon:
            raise ValueError(
                "simulation.horizon_months is shorter than sampled crash windows "
                f"(required at least {required_horizon}, got {configured_horizon})."
            )
        return configured_horizon

    if required_horizon > max_horizon:
        raise ValueError(
            "Sampled crash windows require a horizon larger than "
            f"simulation.max_horizon_months ({required_horizon} > {max_horizon}). "
            "Adjust beliefs or increase max_horizon_months."
        )
    return required_horizon


def _generate_universe_chunk(config, rng, size, horizon_months):
    beliefs = config["beliefs"]
    market = config["market"]
    simulation = config["simulation"]

    prob_crash = _sample_probability(beliefs["prob_crash"], size, rng)
    crash_start = _sample_positive_int(beliefs["crash_start_month"], size, rng, minimum=1)
    crash_duration = _sample_positive_int(beliefs["crash_duration_months"], size, rng, minimum=1)
    crash_severity = np.clip(
        _sample_distribution(beliefs["crash_severity_drop"], size, rng),
        1e-4,
        0.95,
    )
    prob_layoff_during_crash = _sample_probability(beliefs["prob_layoff_during_crash"], size, rng)
    prob_layoff_baseline = _sample_probability(beliefs["prob_layoff_baseline"], size, rng)

    is_crash = rng.random(size) < prob_crash
    crash_end = crash_start + crash_duration
    if np.any(is_crash) and int(np.max(crash_end[is_crash])) > horizon_months:
        raise ValueError(
            "Resolved horizon is shorter than sampled crash windows. "
            "Provide simulation.horizon_months explicitly or increase max_horizon_months."
        )

    down_months = np.maximum(1, crash_duration // 2)
    up_months = np.maximum(1, crash_duration - down_months)
    down_end = crash_start + down_months

    sigma = market["volatility_annual"] / np.sqrt(12.0)
    monthly_market_drift = market["market_yield_annual"] / 12.0
    monthly_returns = rng.normal(0.0, sigma, size=(size, horizon_months)).astype(np.float32)

    if np.any(is_crash):
        mu_down = (1.0 - crash_severity) ** (1.0 / down_months) - 1.0
        mu_up = (1.0 / (1.0 - crash_severity)) ** (1.0 / up_months) - 1.0
        for month in range(1, horizon_months + 1):
            drift = np.full(size, monthly_market_drift, dtype=np.float32)
            mask_down = is_crash & (month >= crash_start) & (month < down_end)
            if np.any(mask_down):
                drift[mask_down] = mu_down[mask_down].astype(np.float32)
            mask_up = is_crash & (month >= down_end) & (month < crash_end)
            if np.any(mask_up):
                drift[mask_up] = mu_up[mask_up].astype(np.float32)
            monthly_returns[:, month - 1] += drift
    else:
        monthly_returns += np.float32(monthly_market_drift)
    np.clip(monthly_returns, -0.95, 1.5, out=monthly_returns)

    layoff_probabilities_are_annual = simulation["layoff_probabilities_are_annual"]
    crash_layoff_min_duration = simulation["crash_layoff_min_duration_months"]
    baseline_layoff_min_duration = simulation["baseline_layoff_min_duration_months"]
    baseline_layoff_max_duration = simulation["baseline_layoff_max_duration_months"]

    layoff_start = np.full(size, horizon_months + 1, dtype=np.int32)
    layoff_end = np.full(size, horizon_months + 1, dtype=np.int32)
    will_lose_job = np.zeros(size, dtype=bool)

    crash_idx = np.flatnonzero(is_crash)
    if crash_idx.size > 0:
        crash_layoff_event_prob = prob_layoff_during_crash[crash_idx]
        if layoff_probabilities_are_annual:
            crash_years = (crash_end[crash_idx] - crash_start[crash_idx]) / 12.0
            crash_layoff_event_prob = 1.0 - np.power(1.0 - crash_layoff_event_prob, crash_years)
            crash_layoff_event_prob = np.clip(crash_layoff_event_prob, 0.0, 1.0)

        crash_layoff = rng.random(crash_idx.size) < crash_layoff_event_prob
        crash_layoff_idx = crash_idx[crash_layoff]
        if crash_layoff_idx.size > 0:
            latest_start = crash_end[crash_layoff_idx] - crash_layoff_min_duration
            eligible = latest_start >= crash_start[crash_layoff_idx]
            eligible_idx = crash_layoff_idx[eligible]

            if eligible_idx.size > 0:
                eligible_latest_start = latest_start[eligible]
                start_window = eligible_latest_start - crash_start[eligible_idx] + 1
                offsets = np.floor(rng.random(eligible_idx.size) * start_window).astype(np.int32)
                starts = crash_start[eligible_idx] + offsets
                max_duration = crash_end[eligible_idx] - starts
                duration_span = max_duration - crash_layoff_min_duration + 1
                duration_offsets = np.floor(rng.random(eligible_idx.size) * duration_span).astype(np.int32)
                durations = crash_layoff_min_duration + duration_offsets
                will_lose_job[eligible_idx] = True
                layoff_start[eligible_idx] = starts
                layoff_end[eligible_idx] = starts + durations

    non_crash_idx = np.flatnonzero(~is_crash)
    if non_crash_idx.size > 0:
        baseline_layoff_event_prob = prob_layoff_baseline[non_crash_idx]
        if layoff_probabilities_are_annual:
            horizon_years = horizon_months / 12.0
            baseline_layoff_event_prob = 1.0 - np.power(1.0 - baseline_layoff_event_prob, horizon_years)
            baseline_layoff_event_prob = np.clip(baseline_layoff_event_prob, 0.0, 1.0)

        baseline_layoff = rng.random(non_crash_idx.size) < baseline_layoff_event_prob
        baseline_layoff_idx = non_crash_idx[baseline_layoff]
        if baseline_layoff_idx.size > 0:
            latest_start = horizon_months - baseline_layoff_min_duration + 1
            if latest_start >= 1:
                starts = rng.integers(1, latest_start + 1, size=baseline_layoff_idx.size, dtype=np.int32)
                max_duration = np.minimum(
                    baseline_layoff_max_duration,
                    horizon_months - starts + 1,
                )
                duration_span = max_duration - baseline_layoff_min_duration + 1
                duration_offsets = np.floor(
                    rng.random(baseline_layoff_idx.size) * duration_span
                ).astype(np.int32)
                durations = baseline_layoff_min_duration + duration_offsets
                will_lose_job[baseline_layoff_idx] = True
                layoff_start[baseline_layoff_idx] = starts
                layoff_end[baseline_layoff_idx] = starts + durations

    months = np.arange(1, horizon_months + 1, dtype=np.int32)[None, :]
    unemployment_matrix = will_lose_job[:, None] & (months >= layoff_start[:, None]) & (
        months < layoff_end[:, None]
    )

    chunk_summary = {
        "n": size,
        "crash_count": int(np.sum(is_crash)),
        "crash_start_sum": float(np.sum(crash_start[is_crash])) if np.any(is_crash) else 0.0,
        "crash_duration_sum": float(np.sum(crash_duration[is_crash])) if np.any(is_crash) else 0.0,
        "crash_severity_sum": float(np.sum(crash_severity[is_crash])) if np.any(is_crash) else 0.0,
        "layoff_crash_count": int(np.sum(will_lose_job[is_crash])) if np.any(is_crash) else 0,
        "non_crash_count": int(np.sum(~is_crash)),
        "layoff_non_crash_count": int(np.sum(will_lose_job[~is_crash])) if np.any(~is_crash) else 0,
    }

    universe_chunk = {
        "n_sims": size,
        "horizon_months": horizon_months,
        "monthly_returns": monthly_returns,
        "unemployment_matrix": unemployment_matrix,
    }
    return universe_chunk, chunk_summary


def _finalize_streaming_summary(summary_acc, horizon_months):
    total = summary_acc["total"]
    crash_count = summary_acc["crash_count"]
    non_crash_count = summary_acc["non_crash_count"]
    return {
        "horizon_months": int(horizon_months),
        "realized_crash_rate": float(crash_count / total),
        "mean_crash_start_if_crash": (
            float(summary_acc["crash_start_sum"] / crash_count) if crash_count else None
        ),
        "mean_crash_duration_if_crash": (
            float(summary_acc["crash_duration_sum"] / crash_count) if crash_count else None
        ),
        "mean_crash_severity_if_crash": (
            float(summary_acc["crash_severity_sum"] / crash_count) if crash_count else None
        ),
        "layoff_rate_if_crash": float(summary_acc["layoff_crash_count"] / crash_count) if crash_count else 0.0,
        "layoff_rate_if_no_crash": (
            float(summary_acc["layoff_non_crash_count"] / non_crash_count) if non_crash_count else 0.0
        ),
    }


def _is_oom_error(exc):
    text = str(exc).lower()
    return (
        "out of memory" in text
        or "cudaerrormemoryallocation" in text
        or "memory allocation" in text
        or "cuda malloc" in text
    )


def _estimate_cuda_working_set_bytes(horizon_months, chunk_size, fraction_tile_size, expected_sample_count):
    # Approximate bytes used by the aggregate CUDA path for one tile evaluation.
    input_returns = chunk_size * horizon_months * 4
    input_unemployment = chunk_size * horizon_months * 1
    sample_map = chunk_size * 4 if expected_sample_count > 0 else 0
    aggregate_sums = fraction_tile_size * (8 + 8 + 8)  # sum_final, sum_log, ruin_count
    sampled_values = fraction_tile_size * expected_sample_count * 8
    base = input_returns + input_unemployment + sample_map + aggregate_sums + sampled_values
    return int(base * 1.40)  # headroom for runtime/temp allocations


def _autotune_streaming_geometry(config, n_fractions, horizon_months, gpu_meta, use_cuda_backend):
    simulation = config["simulation"]
    gpu = config["gpu"]
    n_sims = simulation["n_sims"]
    sample_size = 0
    if gpu["cvar_mode"] == "streaming_near_exact":
        sample_size = min(gpu["sample_size"], n_sims)
    sample_ratio = (sample_size / n_sims) if sample_size > 0 else 0.0

    chunk_size = min(n_sims, gpu["scenario_chunk_size"])
    fraction_tile_size = min(n_fractions, gpu["fraction_tile_size"])
    min_chunk_size = min(n_sims, gpu["min_chunk_size"])

    note = None
    if not use_cuda_backend:
        return chunk_size, fraction_tile_size, note

    free_mb = gpu_meta.get("device_memory_free_mb")
    total_mb = gpu_meta.get("device_memory_total_mb")
    base_mb = free_mb if free_mb is not None else total_mb
    if base_mb is None:
        return chunk_size, fraction_tile_size, note

    budget_bytes = int(base_mb * 1024 * 1024 * gpu["max_vram_utilization"])
    while True:
        expected_sample_count = min(chunk_size, int(np.ceil(chunk_size * sample_ratio))) if sample_ratio > 0 else 0
        if (
            _estimate_cuda_working_set_bytes(
                horizon_months,
                chunk_size,
                fraction_tile_size,
                expected_sample_count,
            )
            <= budget_bytes
        ):
            break
        prev_chunk = chunk_size
        prev_tile = fraction_tile_size
        if fraction_tile_size > 1 and (chunk_size <= min_chunk_size or fraction_tile_size > 4):
            fraction_tile_size = max(1, int(np.ceil(fraction_tile_size / 2)))
        elif chunk_size > min_chunk_size:
            chunk_size = max(min_chunk_size, int(np.floor(chunk_size * gpu["oom_backoff_factor"])))
        if prev_chunk == chunk_size and prev_tile == fraction_tile_size:
            break

    if chunk_size != min(n_sims, gpu["scenario_chunk_size"]) or fraction_tile_size != min(
        n_fractions, gpu["fraction_tile_size"]
    ):
        note = (
            "Auto-tuned streaming geometry from "
            f"(chunk={min(n_sims, gpu['scenario_chunk_size'])}, tile={min(n_fractions, gpu['fraction_tile_size'])}) "
            f"to (chunk={chunk_size}, tile={fraction_tile_size}) based on GPU memory budget."
        )
    return chunk_size, fraction_tile_size, note


def _simulate_fraction_tile_backend(universe_chunk, tile_fractions, config, use_cuda_backend):
    if use_cuda_backend:
        return gpu_backend.simulate_fraction_tile(
            universe_chunk["monthly_returns"],
            universe_chunk["unemployment_matrix"],
            tile_fractions,
            config["portfolio"],
            config["market"],
        )

    chunk_n = universe_chunk["n_sims"]
    tile_n = len(tile_fractions)
    tile_final = np.empty((tile_n, chunk_n), dtype=np.float64)
    tile_ruined = np.empty((tile_n, chunk_n), dtype=bool)
    for idx, fraction in enumerate(tile_fractions):
        final_net_worth, ruined = _simulate_strategy_fraction(fraction, universe_chunk, config)
        tile_final[idx] = final_net_worth
        tile_ruined[idx] = ruined
    return tile_final, tile_ruined, 0.0


def _run_streaming_primary_pass(
    fractions_to_test,
    config,
    horizon_months,
    chunk_size,
    fraction_tile_size,
    use_cuda_backend,
    sample_indices,
):
    simulation = config["simulation"]
    risk = config["risk"]
    n_sims = simulation["n_sims"]
    n_fractions = len(fractions_to_test)
    sample_size = 0 if sample_indices is None else int(sample_indices.size)

    sample_matrix = None
    if sample_size:
        sample_matrix = np.empty((n_fractions, sample_size), dtype=np.float64)

    sum_final = np.zeros(n_fractions, dtype=np.float64)
    sum_log = np.zeros(n_fractions, dtype=np.float64)
    ruin_count = np.zeros(n_fractions, dtype=np.int64)

    summary_acc = {
        "total": 0,
        "crash_count": 0,
        "crash_start_sum": 0.0,
        "crash_duration_sum": 0.0,
        "crash_severity_sum": 0.0,
        "layoff_crash_count": 0,
        "non_crash_count": 0,
        "layoff_non_crash_count": 0,
    }

    kernel_time_ms = 0.0
    transfer_time_ms = 0.0
    t0 = time.perf_counter()

    rng = np.random.default_rng(simulation["seed"])
    sample_ptr = 0
    for start, end in _iter_chunk_ranges(n_sims, chunk_size):
        chunk_n = end - start
        universe_chunk, chunk_summary = _generate_universe_chunk(config, rng, chunk_n, horizon_months)
        summary_acc["total"] += chunk_summary["n"]
        summary_acc["crash_count"] += chunk_summary["crash_count"]
        summary_acc["crash_start_sum"] += chunk_summary["crash_start_sum"]
        summary_acc["crash_duration_sum"] += chunk_summary["crash_duration_sum"]
        summary_acc["crash_severity_sum"] += chunk_summary["crash_severity_sum"]
        summary_acc["layoff_crash_count"] += chunk_summary["layoff_crash_count"]
        summary_acc["non_crash_count"] += chunk_summary["non_crash_count"]
        summary_acc["layoff_non_crash_count"] += chunk_summary["layoff_non_crash_count"]

        local_sample = np.empty(0, dtype=np.int64)
        local_sample_count = 0
        if sample_size:
            local_positions = []
            while sample_ptr < sample_size and sample_indices[sample_ptr] < end:
                local_positions.append(int(sample_indices[sample_ptr] - start))
                sample_ptr += 1
            local_sample = np.asarray(local_positions, dtype=np.int64)
            local_sample_count = int(local_sample.size)

        for tile_start in range(0, n_fractions, fraction_tile_size):
            tile_end = min(tile_start + fraction_tile_size, n_fractions)
            tile_fractions = fractions_to_test[tile_start:tile_end]
            if use_cuda_backend:
                (
                    tile_sum_final,
                    tile_sum_log,
                    tile_ruin_count,
                    tile_sampled_final,
                    tile_kernel_ms,
                    tile_transfer_ms,
                ) = gpu_backend.simulate_fraction_tile_aggregates(
                    universe_chunk["monthly_returns"],
                    universe_chunk["unemployment_matrix"],
                    tile_fractions,
                    config["portfolio"],
                    config["market"],
                    risk["log_utility_wealth_floor"],
                    sample_positions=(local_sample if local_sample_count else None),
                    streams=config["gpu"]["streams"],
                )
                kernel_time_ms += tile_kernel_ms
                transfer_time_ms += tile_transfer_ms
                sum_final[tile_start:tile_end] += tile_sum_final
                sum_log[tile_start:tile_end] += tile_sum_log
                ruin_count[tile_start:tile_end] += tile_ruin_count
                if local_sample_count:
                    sample_matrix[tile_start:tile_end, sample_ptr - local_sample_count : sample_ptr] = tile_sampled_final
            else:
                tile_final, tile_ruined, tile_kernel_ms = _simulate_fraction_tile_backend(
                    universe_chunk,
                    tile_fractions,
                    config,
                    use_cuda_backend,
                )
                kernel_time_ms += tile_kernel_ms
                sum_final[tile_start:tile_end] += np.sum(tile_final, axis=1)
                sum_log[tile_start:tile_end] += np.sum(
                    np.log(np.maximum(tile_final, risk["log_utility_wealth_floor"])),
                    axis=1,
                )
                ruin_count[tile_start:tile_end] += np.sum(tile_ruined, axis=1)

                if local_sample_count:
                    t_transfer = time.perf_counter()
                    sample_matrix[tile_start:tile_end, sample_ptr - local_sample_count : sample_ptr] = tile_final[
                        :,
                        local_sample,
                    ]
                    transfer_time_ms += (time.perf_counter() - t_transfer) * 1000.0

    total_ms = (time.perf_counter() - t0) * 1000.0
    reduction_time_ms = max(0.0, total_ms - kernel_time_ms - transfer_time_ms)
    return {
        "sum_final": sum_final,
        "sum_log": sum_log,
        "ruin_count": ruin_count,
        "sample_matrix": sample_matrix,
        "summary": _finalize_streaming_summary(summary_acc, horizon_months),
        "kernel_time_ms": kernel_time_ms,
        "transfer_time_ms": transfer_time_ms,
        "reduction_time_ms": reduction_time_ms,
    }


def _build_diagnostics_from_metric_arrays(
    config,
    fractions,
    mean_final,
    log_utility_score,
    ruin_probability,
    p10_final_net_worth,
    cvar_shortfall,
):
    risk = config["risk"]
    diagnostics = []
    for idx, fraction in enumerate(fractions):
        cvar_objective_score = float(mean_final[idx] - risk["lambda"] * cvar_shortfall[idx])
        objective_mode = risk["objective_mode"]
        if objective_mode == "cvar_shortfall":
            display_score = cvar_objective_score
        elif objective_mode == "expected_log_utility":
            display_score = float(log_utility_score[idx])
        else:
            display_score = 0.0
        diagnostics.append(
            {
                "expected_final_net_worth": float(mean_final[idx]),
                "p10_final_net_worth": float(p10_final_net_worth[idx]),
                "cvar_shortfall": float(cvar_shortfall[idx]),
                "ruin_probability": float(ruin_probability[idx]),
                "cvar_objective_score": cvar_objective_score,
                "log_utility_score": float(log_utility_score[idx]),
                "score": float(display_score),
                "fraction_sold_today": float(fraction),
            }
        )
    return diagnostics


def _derive_tail_metrics_from_samples(config, sample_matrix):
    if sample_matrix is None:
        raise ValueError("sample_matrix is required for streaming tail estimation.")
    risk = config["risk"]
    portfolio = config["portfolio"]
    shortfall_floor = risk["shortfall_floor"]
    if shortfall_floor is None:
        shortfall_floor = portfolio["initial_portfolio"]

    n_fractions = sample_matrix.shape[0]
    p10 = np.zeros(n_fractions, dtype=np.float64)
    cvar = np.zeros(n_fractions, dtype=np.float64)
    for idx in range(n_fractions):
        sampled_final = sample_matrix[idx]
        p10[idx] = float(np.percentile(sampled_final, 10))
        sampled_shortfall = np.maximum(0.0, shortfall_floor - sampled_final)
        cvar[idx] = _compute_cvar(sampled_shortfall, risk["alpha"])
    return p10, cvar


def _compute_exact_tail_metrics_via_memmap(
    config,
    fractions_to_test,
    fraction_indices,
    horizon_months,
    chunk_size,
    fraction_tile_size,
    use_cuda_backend,
):
    if not fraction_indices:
        return {}

    simulation = config["simulation"]
    risk = config["risk"]
    portfolio = config["portfolio"]
    gpu = config["gpu"]
    n_sims = simulation["n_sims"]
    shortfall_floor = risk["shortfall_floor"]
    if shortfall_floor is None:
        shortfall_floor = portfolio["initial_portfolio"]

    subset_fractions = np.asarray([fractions_to_test[idx] for idx in fraction_indices], dtype=np.float64)
    subset_n = subset_fractions.size
    tile_size = min(fraction_tile_size, subset_n)
    temp_dir = gpu["exact_temp_dir"] or tempfile.gettempdir()

    fd, mmap_path = tempfile.mkstemp(prefix="cpo_exact_tail_", suffix=".dat", dir=temp_dir)
    os.close(fd)
    exact_map = {}
    try:
        values = np.memmap(mmap_path, mode="w+", dtype=np.float64, shape=(subset_n, n_sims))
        rng = np.random.default_rng(simulation["seed"])
        write_pos = 0
        for start, end in _iter_chunk_ranges(n_sims, chunk_size):
            chunk_n = end - start
            universe_chunk, _ = _generate_universe_chunk(config, rng, chunk_n, horizon_months)
            for tile_start in range(0, subset_n, tile_size):
                tile_end = min(tile_start + tile_size, subset_n)
                tile_fractions = subset_fractions[tile_start:tile_end]
                tile_final, _, _ = _simulate_fraction_tile_backend(
                    universe_chunk,
                    tile_fractions,
                    config,
                    use_cuda_backend,
                )
                values[tile_start:tile_end, write_pos : write_pos + chunk_n] = tile_final
            write_pos += chunk_n
        values.flush()

        for local_idx, global_idx in enumerate(fraction_indices):
            row = np.asarray(values[local_idx], dtype=np.float64)
            p10 = float(np.percentile(row, 10))
            shortfall = np.maximum(0.0, shortfall_floor - row)
            cvar = float(_compute_cvar(shortfall, risk["alpha"]))
            exact_map[int(global_idx)] = (p10, cvar)

        del values
    finally:
        try:
            os.remove(mmap_path)
        except OSError:
            pass

    return exact_map


def _apply_exact_tail_metrics(diagnostics, exact_tail_metrics, config):
    risk = config["risk"]
    for idx, metrics in enumerate(diagnostics):
        if idx not in exact_tail_metrics:
            continue
        p10, cvar = exact_tail_metrics[idx]
        metrics["p10_final_net_worth"] = float(p10)
        metrics["cvar_shortfall"] = float(cvar)
        metrics["cvar_objective_score"] = float(
            metrics["expected_final_net_worth"] - risk["lambda"] * metrics["cvar_shortfall"]
        )
        if risk["objective_mode"] == "cvar_shortfall":
            metrics["score"] = metrics["cvar_objective_score"]


def _select_refine_candidate_indices(diagnostics, max_ruin_probability, refine_top_k):
    eligible = [idx for idx, item in enumerate(diagnostics) if item["ruin_probability"] <= max_ruin_probability]
    if not eligible:
        return []
    top_k = min(refine_top_k, len(eligible))
    by_cvar = sorted(eligible, key=lambda idx: diagnostics[idx]["cvar_objective_score"], reverse=True)[:top_k]
    by_log = sorted(eligible, key=lambda idx: diagnostics[idx]["log_utility_score"], reverse=True)[:top_k]
    merged = sorted(set(by_cvar + by_log))
    return merged


def _collect_fraction_metrics_streaming(fractions_to_test, config):
    simulation = config["simulation"]
    gpu = config["gpu"]
    n_sims = simulation["n_sims"]
    n_fractions = len(fractions_to_test)

    use_cuda_backend = bool(gpu["prefer_cuda_extension"])
    fallback_reasons = []
    if use_cuda_backend:
        try:
            gpu_backend.load_cuda_extension()
        except gpu_backend.GpuBackendError as exc:
            use_cuda_backend = False
            fallback_reasons.append(str(exc))
    else:
        fallback_reasons.append("Using numpy_streaming backend because gpu.prefer_cuda_extension is False.")

    gpu_meta = gpu_backend.query_device_metadata(gpu["device_id"])
    horizon_months = _resolve_streaming_horizon(config)
    chunk_size, fraction_tile_size, tune_note = _autotune_streaming_geometry(
        config,
        n_fractions,
        horizon_months,
        gpu_meta,
        use_cuda_backend,
    )
    if tune_note is not None:
        fallback_reasons.append(tune_note)

    min_chunk = min(n_sims, gpu["min_chunk_size"])
    kernel_time_ms = 0.0
    transfer_time_ms = 0.0
    reduction_time_ms = 0.0
    summary = None

    while True:
        try:
            sample_indices = None
            if gpu["cvar_mode"] == "streaming_near_exact":
                sample_size = min(gpu["sample_size"], n_sims)
                sample_indices = _build_sampling_indices(
                    n_sims,
                    sample_size,
                    simulation["seed"] + 10_003,
                )

            primary = _run_streaming_primary_pass(
                fractions_to_test,
                config,
                horizon_months,
                chunk_size,
                fraction_tile_size,
                use_cuda_backend,
                sample_indices,
            )
            kernel_time_ms = primary["kernel_time_ms"]
            transfer_time_ms = primary["transfer_time_ms"]
            reduction_time_ms = primary["reduction_time_ms"]
            summary = primary["summary"]

            mean_final = primary["sum_final"] / n_sims
            log_score = primary["sum_log"] / n_sims
            ruin_probability = primary["ruin_count"] / n_sims

            if gpu["cvar_mode"] == "streaming_near_exact":
                p10, cvar = _derive_tail_metrics_from_samples(config, primary["sample_matrix"])
            else:
                p10 = np.zeros(n_fractions, dtype=np.float64)
                cvar = np.zeros(n_fractions, dtype=np.float64)

            diagnostics = _build_diagnostics_from_metric_arrays(
                config,
                fractions_to_test,
                mean_final,
                log_score,
                ruin_probability,
                p10,
                cvar,
            )

            exact_start = time.perf_counter()
            if gpu["cvar_mode"] == "exact_two_pass":
                all_indices = list(range(n_fractions))
                exact_map = _compute_exact_tail_metrics_via_memmap(
                    config,
                    fractions_to_test,
                    all_indices,
                    horizon_months,
                    chunk_size,
                    fraction_tile_size,
                    use_cuda_backend,
                )
                _apply_exact_tail_metrics(diagnostics, exact_map, config)
            elif gpu["cvar_refine_pass"]:
                refine_indices = _select_refine_candidate_indices(
                    diagnostics,
                    config["risk"]["max_ruin_probability"],
                    gpu["refine_top_k"],
                )
                exact_map = _compute_exact_tail_metrics_via_memmap(
                    config,
                    fractions_to_test,
                    refine_indices,
                    horizon_months,
                    chunk_size,
                    fraction_tile_size,
                    use_cuda_backend,
                )
                _apply_exact_tail_metrics(diagnostics, exact_map, config)
            reduction_time_ms += (time.perf_counter() - exact_start) * 1000.0
            break
        except Exception as exc:
            if not use_cuda_backend or not _is_oom_error(exc):
                raise

            prev_chunk = chunk_size
            prev_tile = fraction_tile_size
            if chunk_size > min_chunk:
                chunk_size = max(min_chunk, int(np.floor(chunk_size * gpu["oom_backoff_factor"])))
            elif fraction_tile_size > 1:
                fraction_tile_size = max(1, int(np.floor(fraction_tile_size * gpu["oom_backoff_factor"])))

            if prev_chunk == chunk_size and prev_tile == fraction_tile_size:
                raise
            fallback_reasons.append(
                "CUDA OOM encountered; backoff applied to "
                f"chunk={chunk_size}, tile={fraction_tile_size}."
            )

    fallback_reason = "; ".join(fallback_reasons) if fallback_reasons else None
    execution = {
        "mode": "parallel",
        "workers_used": int(gpu["streams"]),
        "backend": "cuda" if use_cuda_backend else "numpy_streaming",
        "start_method": None,
        "fraction_tasks": int(n_fractions),
        "fallback_reason": fallback_reason,
        "gpu_name": gpu_meta["gpu_name"],
        "driver_version": gpu_meta["driver_version"],
        "cuda_runtime_version": None,
        "device_memory_total_mb": gpu_meta["device_memory_total_mb"],
        "device_memory_free_mb": gpu_meta.get("device_memory_free_mb"),
        "chunk_size_used": int(chunk_size),
        "fraction_tile_used": int(fraction_tile_size),
        "precision_mode": gpu["precision_mode"],
        "cvar_mode": gpu["cvar_mode"],
        "kernel_time_ms": float(kernel_time_ms),
        "transfer_time_ms": float(transfer_time_ms),
        "reduction_time_ms": float(reduction_time_ms),
    }
    return diagnostics, execution, summary


def _init_parallel_worker(universe, config):
    global _PARALLEL_UNIVERSE
    global _PARALLEL_CONFIG
    _PARALLEL_UNIVERSE = universe
    _PARALLEL_CONFIG = config


def _evaluate_fraction_task(task):
    idx, fraction = task

    if _PARALLEL_UNIVERSE is None or _PARALLEL_CONFIG is None:
        raise RuntimeError("Parallel worker is not initialized.")

    final_net_worth, ruined = _simulate_strategy_fraction(fraction, _PARALLEL_UNIVERSE, _PARALLEL_CONFIG)
    metrics = _score_strategy(final_net_worth, ruined, _PARALLEL_CONFIG)
    metrics["fraction_sold_today"] = float(fraction)
    return idx, metrics


def _evaluate_fraction_direct(idx, fraction, universe, config):
    final_net_worth, ruined = _simulate_strategy_fraction(fraction, universe, config)
    metrics = _score_strategy(final_net_worth, ruined, config)
    metrics["fraction_sold_today"] = float(fraction)
    return idx, metrics


def _default_parallel_start_method():
    methods = multiprocessing.get_all_start_methods()
    if os.name == "posix" and "fork" in methods:
        return "fork"
    if "spawn" in methods:
        return "spawn"
    return methods[0]


def _resolve_execution_settings(config, task_count):
    simulation = config["simulation"]
    resolved_workers = simulation["parallel_workers"]
    if resolved_workers is None:
        resolved_workers = os.cpu_count() or 1

    execution = {
        "mode": "single",
        "workers_used": 1,
        "backend": "single",
        "start_method": None,
        "fraction_tasks": task_count,
        "fallback_reason": None,
        "gpu_name": None,
        "driver_version": None,
        "cuda_runtime_version": None,
        "device_memory_total_mb": None,
        "device_memory_free_mb": None,
        "chunk_size_used": None,
        "fraction_tile_used": None,
        "precision_mode": None,
        "cvar_mode": None,
        "kernel_time_ms": None,
        "transfer_time_ms": None,
        "reduction_time_ms": None,
    }

    if not simulation["parallel_enabled"]:
        return execution
    if task_count < simulation["parallel_min_fraction_tasks"]:
        return execution

    workers_used = max(1, min(resolved_workers, task_count))
    if workers_used <= 1:
        return execution

    execution["mode"] = "parallel"
    execution["workers_used"] = workers_used
    execution["backend"] = "multiprocessing"
    execution["start_method"] = (
        simulation["parallel_start_method"] or _default_parallel_start_method()
    )
    return execution


def _collect_fraction_metrics_single_thread(fractions_to_test, universe, config):
    diagnostics = []
    for idx, fraction in enumerate(fractions_to_test):
        _, metrics = _evaluate_fraction_direct(idx, fraction, universe, config)
        diagnostics.append(metrics)
    return diagnostics


def _collect_fraction_metrics_parallel(fractions_to_test, universe, config, execution):
    tasks = [(idx, float(fraction)) for idx, fraction in enumerate(fractions_to_test)]
    futures = {}
    results = {}
    mp_context = multiprocessing.get_context(execution["start_method"])

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=execution["workers_used"],
        mp_context=mp_context,
        initializer=_init_parallel_worker,
        initargs=(universe, config),
    ) as executor:
        for task in tasks:
            future = executor.submit(_evaluate_fraction_task, task)
            futures[future] = task[0]

        for future in concurrent.futures.as_completed(futures):
            idx, metrics = future.result()
            results[idx] = metrics

    return [results[idx] for idx in sorted(results)]


def _collect_fraction_metrics(fractions_to_test, universe, config):
    execution = _resolve_execution_settings(config, len(fractions_to_test))
    if execution["mode"] == "parallel":
        try:
            diagnostics = _collect_fraction_metrics_parallel(
                fractions_to_test,
                universe,
                config,
                execution,
            )
        except Exception as exc:
            execution["mode"] = "single"
            execution["workers_used"] = 1
            execution["backend"] = "single"
            execution["start_method"] = None
            execution["fallback_reason"] = str(exc)
            diagnostics = _collect_fraction_metrics_single_thread(
                fractions_to_test,
                universe,
                config,
            )
    else:
        diagnostics = _collect_fraction_metrics_single_thread(
            fractions_to_test,
            universe,
            config,
        )
    return diagnostics, execution


def _eligible_indices_for_guardrail(diagnostics, max_ruin_probability):
    eligible_indices = [
        idx for idx, item in enumerate(diagnostics) if item["ruin_probability"] <= max_ruin_probability
    ]
    if not eligible_indices:
        raise ValueError(
            "No strategy satisfies the ruin probability guardrail. "
            f"Increase risk.max_ruin_probability (currently {max_ruin_probability:.2%}) or adjust assumptions."
        )
    return eligible_indices


def _select_best_strategy(diagnostics, objective_mode, max_ruin_probability):
    objective_key = _objective_key_for_mode(objective_mode)
    eligible_indices = _eligible_indices_for_guardrail(diagnostics, max_ruin_probability)

    sorted_indices = sorted(
        eligible_indices,
        key=lambda idx: diagnostics[idx][objective_key],
        reverse=True,
    )

    best_idx = sorted_indices[0]
    second_idx = sorted_indices[1] if len(sorted_indices) > 1 else best_idx

    left_delta = None
    right_delta = None
    best_pos = eligible_indices.index(best_idx)
    if best_pos > 0:
        left_idx = eligible_indices[best_pos - 1]
        left_delta = diagnostics[best_idx][objective_key] - diagnostics[left_idx][objective_key]
    if best_pos < len(eligible_indices) - 1:
        right_idx = eligible_indices[best_pos + 1]
        right_delta = diagnostics[best_idx][objective_key] - diagnostics[right_idx][objective_key]

    local_margin = None
    if left_delta is not None and right_delta is not None:
        local_margin = min(left_delta, right_delta)
    elif left_delta is not None:
        local_margin = left_delta
    elif right_delta is not None:
        local_margin = right_delta

    best = diagnostics[best_idx]
    second_best = diagnostics[second_idx]
    top_candidates = [diagnostics[idx] for idx in sorted_indices[:5]]

    return {
        "objective_mode": objective_mode,
        "objective_key": objective_key,
        "objective_score": best[objective_key],
        "recommended_fraction": best["fraction_sold_today"],
        "expected_final_net_worth": best["expected_final_net_worth"],
        "cvar_shortfall": best["cvar_shortfall"],
        "ruin_probability": best["ruin_probability"],
        "log_utility_score": best["log_utility_score"],
        "cvar_objective_score": best["cvar_objective_score"],
        "score_gap_vs_second_best": best[objective_key] - second_best[objective_key],
        "local_score_margin": local_margin,
        "eligible_strategies": len(eligible_indices),
        "top_candidates": top_candidates,
    }


def _normalized_regret(best_value, candidate_value):
    scale = max(abs(best_value), 1.0)
    return max(0.0, (best_value - candidate_value) / scale)


def _select_consensus_strategy(diagnostics, max_ruin_probability, consensus_tolerance):
    objective_mode = "consensus"
    objective_key = _objective_key_for_mode(objective_mode)
    eligible_indices = _eligible_indices_for_guardrail(diagnostics, max_ruin_probability)

    best_cvar = max(diagnostics[idx]["cvar_objective_score"] for idx in eligible_indices)
    best_log = max(diagnostics[idx]["log_utility_score"] for idx in eligible_indices)

    enriched = []
    for idx in eligible_indices:
        item = diagnostics[idx]
        cvar_regret = _normalized_regret(best_cvar, item["cvar_objective_score"])
        log_regret = _normalized_regret(best_log, item["log_utility_score"])
        max_regret = max(cvar_regret, log_regret)
        sum_regret = cvar_regret + log_regret
        consensus_pass = cvar_regret <= consensus_tolerance and log_regret <= consensus_tolerance
        consensus_score = -max_regret
        enriched.append(
            {
                "idx": idx,
                "cvar_regret": cvar_regret,
                "log_regret": log_regret,
                "max_regret": max_regret,
                "sum_regret": sum_regret,
                "consensus_pass": consensus_pass,
                "consensus_score": consensus_score,
            }
        )

    pass_pool = [entry for entry in enriched if entry["consensus_pass"]]
    used_fallback = False
    candidate_pool = pass_pool
    if not candidate_pool:
        candidate_pool = enriched
        used_fallback = True

    sorted_entries = sorted(
        candidate_pool,
        key=lambda entry: (
            entry["max_regret"],
            entry["sum_regret"],
            -diagnostics[entry["idx"]]["cvar_objective_score"],
            -diagnostics[entry["idx"]]["log_utility_score"],
        ),
    )

    best_entry = sorted_entries[0]
    second_entry = sorted_entries[1] if len(sorted_entries) > 1 else best_entry

    best_idx = best_entry["idx"]
    best = diagnostics[best_idx]
    second_best = diagnostics[second_entry["idx"]]
    top_candidates = [diagnostics[entry["idx"]] for entry in sorted_entries[:5]]

    return {
        "objective_mode": objective_mode,
        "objective_key": objective_key,
        "objective_score": best_entry["consensus_score"],
        "recommended_fraction": best["fraction_sold_today"],
        "expected_final_net_worth": best["expected_final_net_worth"],
        "cvar_shortfall": best["cvar_shortfall"],
        "ruin_probability": best["ruin_probability"],
        "log_utility_score": best["log_utility_score"],
        "cvar_objective_score": best["cvar_objective_score"],
        "score_gap_vs_second_best": best_entry["consensus_score"] - second_entry["consensus_score"],
        "local_score_margin": None,
        "eligible_strategies": len(eligible_indices),
        "top_candidates": top_candidates,
        "consensus_tolerance": consensus_tolerance,
        "consensus_within_tolerance_count": len(pass_pool),
        "consensus_used_fallback": used_fallback,
        "consensus_max_regret": best_entry["max_regret"],
        "consensus_cvar_regret": best_entry["cvar_regret"],
        "consensus_log_regret": best_entry["log_regret"],
    }


def run_monte_carlo_optimization(config=None, verbose=True):
    merged_config = _deep_merge(DEFAULT_CONFIG, config or {})
    validate_config(merged_config)

    decision_grid = merged_config["decision_grid"]
    risk = merged_config["risk"]
    simulation = merged_config["simulation"]
    gpu = merged_config["gpu"]

    fractions_to_test = np.linspace(
        decision_grid["min_fraction"],
        decision_grid["max_fraction"],
        decision_grid["num_points"],
    )

    primary_objective_mode = risk["objective_mode"]
    objective_label = _objective_label(primary_objective_mode)
    objective_label_for_table = objective_label
    if primary_objective_mode == "consensus":
        objective_label_for_table = "Mean - lambda*CVaR"

    gpu_error = None
    if gpu["enabled"]:
        try:
            diagnostics, execution, universe_summary = _collect_fraction_metrics_streaming(
                fractions_to_test,
                merged_config,
            )
            n_sims_for_output = simulation["n_sims"]
        except Exception as exc:
            if not gpu["fallback_to_cpu_on_error"]:
                raise
            gpu_error = str(exc)
            try:
                streaming_fallback_cfg = _deep_merge(
                    merged_config,
                    {"gpu": {"enabled": True, "prefer_cuda_extension": False}},
                )
                diagnostics, execution, universe_summary = _collect_fraction_metrics_streaming(
                    fractions_to_test,
                    streaming_fallback_cfg,
                )
                n_sims_for_output = simulation["n_sims"]
            except Exception:
                universe = _generate_universe(merged_config)
                diagnostics, execution = _collect_fraction_metrics(
                    fractions_to_test,
                    universe,
                    merged_config,
                )
                universe_summary = universe["summary"]
                n_sims_for_output = universe["n_sims"]
    else:
        universe = _generate_universe(merged_config)
        diagnostics, execution = _collect_fraction_metrics(
            fractions_to_test,
            universe,
            merged_config,
        )
        universe_summary = universe["summary"]
        n_sims_for_output = universe["n_sims"]

    if gpu_error is not None:
        if execution["fallback_reason"] is None:
            execution["fallback_reason"] = gpu_error
        else:
            execution["fallback_reason"] = f"{gpu_error}; {execution['fallback_reason']}"

    if verbose:
        summary = universe_summary
        cvar_label = f"CVaR{int(risk['alpha'] * 100)} Shortfall"
        print(f"Running {n_sims_for_output:,} belief-weighted simulations...\n")
        print("Universe summary:")
        print(f"  Horizon months:              {summary['horizon_months']}")
        print(f"  Realized crash rate:         {summary['realized_crash_rate']:.1%}")
        if summary["mean_crash_start_if_crash"] is not None:
            print(f"  Avg crash start (if crash):  {summary['mean_crash_start_if_crash']:.1f} months")
            print(
                f"  Avg crash duration:          {summary['mean_crash_duration_if_crash']:.1f} months"
            )
            print(
                f"  Avg crash severity drop:     {summary['mean_crash_severity_if_crash']:.1%}"
            )
        print(f"  Layoff rate if crash:        {summary['layoff_rate_if_crash']:.1%}")
        print(f"  Layoff rate if no crash:     {summary['layoff_rate_if_no_crash']:.1%}")
        print()
        print(
            f"{'Portion Sold Today':>20} | "
            f"{'Expected Final Wealth':>24} | "
            f"{cvar_label:>17} | "
            f"{objective_label_for_table:>16}"
        )
        print("-" * 92)
        for metrics in diagnostics:
            fraction = metrics["fraction_sold_today"]
            if round(fraction * 100) % 10 == 0:
                objective_display_value = metrics["score"]
                objective_display_mode = primary_objective_mode
                if primary_objective_mode == "consensus":
                    objective_display_value = metrics["cvar_objective_score"]
                    objective_display_mode = "cvar_shortfall"
                objective_display = _format_objective_for_table(objective_display_value, objective_display_mode)
                print(
                    f"{fraction * 100:19.0f}% | "
                    f"${metrics['expected_final_net_worth']:23,.0f} | "
                    f"${metrics['cvar_shortfall']:16,.0f} | "
                    f"{objective_display}"
                )

    cvar_selection = _select_best_strategy(
        diagnostics,
        "cvar_shortfall",
        risk["max_ruin_probability"],
    )
    log_selection = _select_best_strategy(
        diagnostics,
        "expected_log_utility",
        risk["max_ruin_probability"],
    )
    consensus_selection = _select_consensus_strategy(
        diagnostics,
        risk["max_ruin_probability"],
        risk["consensus_tolerance"],
    )

    mode_to_selection = {
        "cvar_shortfall": cvar_selection,
        "expected_log_utility": log_selection,
        "consensus": consensus_selection,
    }
    primary_selection = mode_to_selection[primary_objective_mode]
    secondary_modes = [mode for mode in ("cvar_shortfall", "expected_log_utility", "consensus") if mode != primary_objective_mode]

    result = {
        "recommended_fraction": primary_selection["recommended_fraction"],
        "objective_score": primary_selection["objective_score"],
        "expected_final_net_worth": primary_selection["expected_final_net_worth"],
        "cvar_shortfall": primary_selection["cvar_shortfall"],
        "ruin_probability": primary_selection["ruin_probability"],
        "score_gap_vs_second_best": primary_selection["score_gap_vs_second_best"],
        "local_score_margin": primary_selection["local_score_margin"],
        "top_candidates": primary_selection["top_candidates"],
        "primary_objective_mode": primary_objective_mode,
        "max_ruin_probability": risk["max_ruin_probability"],
        "objective_comparison": mode_to_selection,
        "diagnostics_by_fraction": diagnostics,
        "universe_summary": universe_summary,
        "execution": execution,
    }

    if verbose:
        print("-" * 92)
        print(
            f"Execution mode:                        {execution['mode']} "
            f"({execution['backend']}, workers={execution['workers_used']})"
        )
        if execution["start_method"] is not None:
            print(f"Parallel start method:                 {execution['start_method']}")
        if execution["fallback_reason"] is not None:
            print(f"Backend fallback reason:               {execution['fallback_reason']}")
        if execution["gpu_name"] is not None:
            print(f"GPU device:                            {execution['gpu_name']}")
        if execution["driver_version"] is not None:
            print(f"GPU driver version:                    {execution['driver_version']}")
        if execution["device_memory_total_mb"] is not None:
            print(f"GPU memory total (MB):                 {execution['device_memory_total_mb']}")
        if execution["device_memory_free_mb"] is not None:
            print(f"GPU memory free (MB):                  {execution['device_memory_free_mb']}")
        if execution["chunk_size_used"] is not None:
            print(f"Scenario chunk size:                   {execution['chunk_size_used']:,}")
        if execution["fraction_tile_used"] is not None:
            print(f"Fraction tile size:                    {execution['fraction_tile_used']}")
        if execution["precision_mode"] is not None:
            print(f"GPU precision mode:                    {execution['precision_mode']}")
        if execution["cvar_mode"] is not None:
            print(f"CVaR mode:                             {execution['cvar_mode']}")
        if execution["kernel_time_ms"] is not None:
            print(f"Kernel time (ms):                      {execution['kernel_time_ms']:.1f}")
        if execution["transfer_time_ms"] is not None:
            print(f"Transfer time (ms):                    {execution['transfer_time_ms']:.1f}")
        if execution["reduction_time_ms"] is not None:
            print(f"Reduction time (ms):                   {execution['reduction_time_ms']:.1f}")
        print(f"Primary objective mode:                {primary_objective_mode}")
        print(f"Ruin probability guardrail:            <= {risk['max_ruin_probability']:.2%}")
        print(f"Recommended sell-today fraction:        {result['recommended_fraction'] * 100:.1f}%")
        print(f"Expected final net worth:              ${result['expected_final_net_worth']:,.0f}")
        print(
            f"CVaR shortfall (worst {int(risk['alpha'] * 100)}%): "
            f"${result['cvar_shortfall']:,.0f}"
        )
        print(f"Ruin probability:                       {result['ruin_probability']:.2%}")
        if primary_objective_mode == "consensus":
            print(
                "Consensus max regret:                  "
                f"{primary_selection['consensus_max_regret']:.6%}"
            )
            print(
                "Consensus tolerance:                   "
                f"{primary_selection['consensus_tolerance']:.6%}"
            )
            print(
                "Consensus set size (feasible):         "
                f"{primary_selection['consensus_within_tolerance_count']}"
            )
            if primary_selection["consensus_used_fallback"]:
                print("Consensus fallback used:               yes (picked closest compromise)")
            else:
                print("Consensus fallback used:               no")
            print(
                "Consensus winner margin vs second:     "
                f"{result['score_gap_vs_second_best']:.6%} consensus-score gap"
            )
        else:
            print(
                "Objective score:                       "
                f"{_format_objective_for_summary(result['objective_score'], primary_objective_mode)}"
            )
            print(
                "Score gap vs second best:              "
                f"{_format_objective_for_summary(result['score_gap_vs_second_best'], primary_objective_mode)}"
            )
            if result["local_score_margin"] is not None:
                print(
                    "Local score margin around optimum:     "
                    f"{_format_objective_for_summary(result['local_score_margin'], primary_objective_mode)}"
                )
        print()
        print("Secondary objective cross-checks:")
        for mode in secondary_modes:
            secondary_selection = mode_to_selection[mode]
            print(f"  Mode:                                {mode}")
            print(
                "  Recommended sell-today fraction:     "
                f"{secondary_selection['recommended_fraction'] * 100:.1f}%"
            )
            print(
                "  Objective score:                     "
                f"{_format_objective_for_summary(secondary_selection['objective_score'], mode)}"
            )
        print()
        print(
            "Decision framing: based on your belief distributions and risk settings, "
            f"selling about {result['recommended_fraction'] * 100:.1f}% today is the "
            "best one-time action under the configured objective and guardrail."
        )

    return result


if __name__ == "__main__":
    run_monte_carlo_optimization()
