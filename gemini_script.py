from copy import deepcopy

import numpy as np


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
        "n_sims": 500_000,
        "seed": 42,
        "min_horizon_months": 72,
        "horizon_months": None,
        "max_horizon_months": 240,
        "post_crash_buffer_months": 24,
        "layoff_probabilities_are_annual": True,
        "crash_layoff_min_duration_months": 24,
        "baseline_layoff_min_duration_months": 1,
        "baseline_layoff_max_duration_months": 12,
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
}

SUPPORTED_DISTRIBUTIONS = {"beta", "triangular", "uniform", "lognormal"}
SUPPORTED_OBJECTIVES = {"cvar_shortfall", "expected_log_utility", "consensus"}


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
    for key in ("portfolio", "market", "simulation", "decision_grid", "risk", "beliefs"):
        if key not in config:
            raise ValueError(f"Missing top-level config section '{key}'.")

    portfolio = config["portfolio"]
    market = config["market"]
    simulation = config["simulation"]
    decision_grid = config["decision_grid"]
    risk = config["risk"]
    beliefs = config["beliefs"]

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

    universe = _generate_universe(merged_config)
    decision_grid = merged_config["decision_grid"]
    risk = merged_config["risk"]

    fractions_to_test = np.linspace(
        decision_grid["min_fraction"],
        decision_grid["max_fraction"],
        decision_grid["num_points"],
    )

    diagnostics = []

    primary_objective_mode = risk["objective_mode"]
    objective_label = _objective_label(primary_objective_mode)
    objective_label_for_table = objective_label
    if primary_objective_mode == "consensus":
        objective_label_for_table = "Mean - lambda*CVaR"

    if verbose:
        summary = universe["summary"]
        cvar_label = f"CVaR{int(risk['alpha'] * 100)} Shortfall"
        print(f"Running {universe['n_sims']:,} belief-weighted simulations...\n")
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

    for fraction in fractions_to_test:
        final_net_worth, ruined = _simulate_strategy_fraction(fraction, universe, merged_config)
        metrics = _score_strategy(final_net_worth, ruined, merged_config)
        metrics["fraction_sold_today"] = float(fraction)
        diagnostics.append(metrics)

        if verbose and round(fraction * 100) % 10 == 0:
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
        "universe_summary": universe["summary"],
    }

    if verbose:
        print("-" * 92)
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
