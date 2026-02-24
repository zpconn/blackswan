import math
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import blackswan as optimizer


def _base_config():
    return {
        "simulation": {
            "n_sims": 1_500,
            "seed": 123,
            "parallel_enabled": False,
        },
        "gpu": {
            "enabled": False,
        },
        "decision_grid": {
            "num_points": 9,
        },
    }


def _deterministic_universe(horizon_months, monthly_return=0.0, unemployed=False):
    return {
        "n_sims": 1,
        "horizon_months": horizon_months,
        "monthly_returns": np.full((1, horizon_months), monthly_return, dtype=np.float64),
        "unemployment_matrix": np.full((1, horizon_months), unemployed, dtype=bool),
    }


def _custom_universe(monthly_returns, unemployed=False):
    monthly_returns = np.asarray(monthly_returns, dtype=np.float64).reshape(1, -1)
    return {
        "n_sims": 1,
        "horizon_months": int(monthly_returns.shape[1]),
        "monthly_returns": monthly_returns,
        "unemployment_matrix": np.full((1, monthly_returns.shape[1]), unemployed, dtype=bool),
    }


def _assert_core_result_shape(result, expected_points):
    assert "recommended_fraction" in result
    assert "objective_score" in result
    assert "diagnostics_by_fraction" in result
    assert "execution" in result
    assert len(result["diagnostics_by_fraction"]) == expected_points
    assert math.isfinite(result["expected_final_net_worth"])
    assert math.isfinite(result["cvar_shortfall"])
    assert 0.0 <= result["ruin_probability"] <= 1.0


def _assert_close_metrics(left, right):
    assert left["recommended_fraction"] == right["recommended_fraction"]
    assert math.isclose(
        left["objective_score"],
        right["objective_score"],
        rel_tol=1e-9,
        abs_tol=1e-7,
    )
    assert math.isclose(
        left["expected_final_net_worth"],
        right["expected_final_net_worth"],
        rel_tol=1e-9,
        abs_tol=1e-4,
    )
    assert math.isclose(
        left["cvar_shortfall"],
        right["cvar_shortfall"],
        rel_tol=1e-9,
        abs_tol=1e-4,
    )
    assert math.isclose(
        left["ruin_probability"],
        right["ruin_probability"],
        rel_tol=1e-9,
        abs_tol=1e-12,
    )


@pytest.mark.parametrize(
    "override",
    [
        {"simulation": {"parallel_workers": 0}},
        {"simulation": {"parallel_enabled": "yes"}},
        {"simulation": {"parallel_start_method": "not_a_method"}},
    ],
)
def test_validate_config_rejects_invalid_parallel_settings(override):
    config = optimizer._deep_merge(optimizer.DEFAULT_CONFIG, override)
    with pytest.raises(ValueError):
        optimizer.validate_config(config)


@pytest.mark.parametrize(
    "override",
    [
        {"gpu": {"fraction_tile_size": 0}},
        {"gpu": {"scenario_chunk_size": 0}},
        {"gpu": {"precision_mode": "bad_mode"}},
        {"gpu": {"cvar_mode": "bad_mode"}},
        {"gpu": {"reduction_workers": 0}},
        {"gpu": {"reduction_workers": True}},
    ],
)
def test_validate_config_rejects_invalid_gpu_settings(override):
    config = optimizer._deep_merge(optimizer.DEFAULT_CONFIG, override)
    with pytest.raises(ValueError):
        optimizer.validate_config(config)


@pytest.mark.parametrize(
    "override,match",
    [
        (
            {"portfolio": {"retirement": {"enabled": True, "start_month_from_start": None}}},
            "start_month_from_start is required",
        ),
        (
            {"portfolio": {"retirement": {"start_month_from_start": -1}}},
            "start_month_from_start must be >= 0",
        ),
        (
            {"portfolio": {"retirement": {"expense_reduction_fraction": 1.2}}},
            r"expense_reduction_fraction must be in \[0, 1\]",
        ),
        (
            {"portfolio": {"retirement": {"dynamic_safe_withdrawal_rate": "yes"}}},
            "dynamic_safe_withdrawal_rate must be a bool",
        ),
    ],
)
def test_validate_config_rejects_invalid_retirement_settings(override, match):
    config = optimizer._deep_merge(optimizer.DEFAULT_CONFIG, override)
    with pytest.raises(ValueError, match=match):
        optimizer.validate_config(config)


@pytest.mark.parametrize(
    "override,match",
    [
        (
            {"portfolio": {"crash_reinvestment": {"crash_drawdown_threshold": 0.0}}},
            r"crash_drawdown_threshold must be in \(0, 1\)",
        ),
        (
            {"portfolio": {"crash_reinvestment": {"recovery_fraction_of_peak": 1.2}}},
            r"recovery_fraction_of_peak must be in \(0, 1\]",
        ),
        (
            {"portfolio": {"crash_reinvestment": {"reinvest_fraction_of_initial_sale_proceeds": 1.2}}},
            r"reinvest_fraction_of_initial_sale_proceeds must be in \[0, 1\]",
        ),
        (
            {"portfolio": {"crash_reinvestment": {"cash_buffer_months": True}}},
            r"cash_buffer_months must be an int >= 0",
        ),
    ],
)
def test_validate_config_rejects_invalid_crash_reinvestment_settings(override, match):
    config = optimizer._deep_merge(optimizer.DEFAULT_CONFIG, override)
    with pytest.raises(ValueError, match=match):
        optimizer.validate_config(config)


def test_retirement_forces_permanent_unemployment_after_start():
    universe = _deterministic_universe(horizon_months=3, monthly_return=0.0, unemployed=False)
    base = optimizer._deep_merge(
        optimizer.DEFAULT_CONFIG,
        {
            "portfolio": {
                "initial_portfolio": 1_000.0,
                "initial_basis": 1_000.0,
                "ltcg_tax_rate": 0.0,
                "monthly_expenses": 0.0,
                "monthly_savings": 100.0,
            },
            "market": {"cash_yield_annual": 0.0},
        },
    )
    no_retirement_cfg = optimizer._deep_merge(base, {"portfolio": {"retirement": {"enabled": False}}})
    retirement_cfg = optimizer._deep_merge(
        base,
        {
            "portfolio": {
                "retirement": {
                    "enabled": True,
                    "start_month_from_start": 1,
                    "safe_withdrawal_rate_annual": 0.0,
                    "expense_reduction_fraction": 1.0,
                    "dynamic_safe_withdrawal_rate": False,
                }
            }
        },
    )

    baseline_final, baseline_ruined = optimizer._simulate_strategy_fraction(0.0, universe, no_retirement_cfg)
    retired_final, retired_ruined = optimizer._simulate_strategy_fraction(0.0, universe, retirement_cfg)

    assert not baseline_ruined[0]
    assert not retired_ruined[0]
    assert math.isclose(baseline_final[0], 1_300.0, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(retired_final[0], 1_100.0, rel_tol=0.0, abs_tol=1e-9)


def test_retirement_applies_expenses_every_month_after_start():
    universe = _deterministic_universe(horizon_months=3, monthly_return=0.0, unemployed=False)
    base = optimizer._deep_merge(
        optimizer.DEFAULT_CONFIG,
        {
            "portfolio": {
                "initial_portfolio": 120.0,
                "initial_basis": 120.0,
                "ltcg_tax_rate": 0.0,
                "monthly_expenses": 50.0,
                "monthly_savings": 0.0,
            },
            "market": {"cash_yield_annual": 0.0},
        },
    )
    no_retirement_cfg = optimizer._deep_merge(base, {"portfolio": {"retirement": {"enabled": False}}})
    retirement_cfg = optimizer._deep_merge(
        base,
        {
            "portfolio": {
                "retirement": {
                    "enabled": True,
                    "start_month_from_start": 1,
                    "safe_withdrawal_rate_annual": 0.0,
                    "expense_reduction_fraction": 1.0,
                    "dynamic_safe_withdrawal_rate": False,
                }
            }
        },
    )

    baseline_final, baseline_ruined = optimizer._simulate_strategy_fraction(1.0, universe, no_retirement_cfg)
    retired_final, retired_ruined = optimizer._simulate_strategy_fraction(1.0, universe, retirement_cfg)

    assert not baseline_ruined[0]
    assert not retired_ruined[0]
    assert math.isclose(baseline_final[0], 120.0, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(retired_final[0], 20.0, rel_tol=0.0, abs_tol=1e-9)


def test_dynamic_retirement_withdrawal_matches_reduced_expenses():
    universe = _deterministic_universe(horizon_months=2, monthly_return=0.0, unemployed=False)
    fixed_cfg = optimizer._deep_merge(
        optimizer.DEFAULT_CONFIG,
        {
            "portfolio": {
                "initial_portfolio": 1_000.0,
                "initial_basis": 1_000.0,
                "ltcg_tax_rate": 0.0,
                "monthly_expenses": 100.0,
                "monthly_savings": 0.0,
                "retirement": {
                    "enabled": True,
                    "start_month_from_start": 0,
                    "safe_withdrawal_rate_annual": 1.2,
                    "expense_reduction_fraction": 0.7,
                    "dynamic_safe_withdrawal_rate": False,
                },
            },
            "market": {"cash_yield_annual": 0.12},
        },
    )
    dynamic_cfg = optimizer._deep_merge(
        fixed_cfg,
        {"portfolio": {"retirement": {"dynamic_safe_withdrawal_rate": True}}},
    )

    fixed_final, fixed_ruined = optimizer._simulate_strategy_fraction(0.0, universe, fixed_cfg)
    dynamic_final, dynamic_ruined = optimizer._simulate_strategy_fraction(0.0, universe, dynamic_cfg)

    assert not fixed_ruined[0]
    assert not dynamic_ruined[0]
    assert math.isclose(fixed_final[0], 860.3, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(dynamic_final[0], 860.0, rel_tol=0.0, abs_tol=1e-6)
    assert dynamic_final[0] < fixed_final[0]


def test_crash_reinvestment_triggers_after_recovery():
    universe = _custom_universe([-0.30, 0.20, 0.10, 0.10], unemployed=False)
    base = optimizer._deep_merge(
        optimizer.DEFAULT_CONFIG,
        {
            "portfolio": {
                "initial_portfolio": 100.0,
                "initial_basis": 100.0,
                "ltcg_tax_rate": 0.0,
                "monthly_expenses": 0.0,
                "monthly_savings": 0.0,
                "crash_reinvestment": {
                    "enabled": True,
                    "crash_drawdown_threshold": 0.20,
                    "recovery_fraction_of_peak": 0.90,
                    "reinvest_fraction_of_initial_sale_proceeds": 1.0,
                    "cash_buffer_months": 0,
                },
            },
            "market": {"cash_yield_annual": 0.0},
        },
    )
    disabled = optimizer._deep_merge(base, {"portfolio": {"crash_reinvestment": {"enabled": False}}})

    with_reinvest, ruined_with = optimizer._simulate_strategy_fraction(1.0, universe, base)
    without_reinvest, ruined_without = optimizer._simulate_strategy_fraction(1.0, universe, disabled)

    assert not ruined_with[0]
    assert not ruined_without[0]
    assert with_reinvest[0] > without_reinvest[0] + 5.0


def test_crash_reinvestment_does_not_trigger_without_required_drawdown():
    universe = _custom_universe([-0.10, 0.05, 0.05, 0.05], unemployed=False)
    base = optimizer._deep_merge(
        optimizer.DEFAULT_CONFIG,
        {
            "portfolio": {
                "initial_portfolio": 100.0,
                "initial_basis": 100.0,
                "ltcg_tax_rate": 0.0,
                "monthly_expenses": 0.0,
                "monthly_savings": 0.0,
                "crash_reinvestment": {
                    "enabled": True,
                    "crash_drawdown_threshold": 0.20,
                    "recovery_fraction_of_peak": 0.90,
                    "reinvest_fraction_of_initial_sale_proceeds": 1.0,
                    "cash_buffer_months": 0,
                },
            },
            "market": {"cash_yield_annual": 0.0},
        },
    )
    disabled = optimizer._deep_merge(base, {"portfolio": {"crash_reinvestment": {"enabled": False}}})

    with_reinvest, _ = optimizer._simulate_strategy_fraction(1.0, universe, base)
    without_reinvest, _ = optimizer._simulate_strategy_fraction(1.0, universe, disabled)

    assert math.isclose(with_reinvest[0], without_reinvest[0], rel_tol=0.0, abs_tol=1e-9)


def test_crash_reinvestment_respects_cash_buffer():
    universe = _custom_universe([-0.30, 0.20, 0.10, 0.10], unemployed=False)
    base = optimizer._deep_merge(
        optimizer.DEFAULT_CONFIG,
        {
            "portfolio": {
                "initial_portfolio": 100.0,
                "initial_basis": 100.0,
                "ltcg_tax_rate": 0.0,
                "monthly_expenses": 20.0,
                "monthly_savings": 0.0,
                "crash_reinvestment": {
                    "enabled": True,
                    "crash_drawdown_threshold": 0.20,
                    "recovery_fraction_of_peak": 0.90,
                    "reinvest_fraction_of_initial_sale_proceeds": 1.0,
                    "cash_buffer_months": 10,
                },
            },
            "market": {"cash_yield_annual": 0.0},
        },
    )
    disabled = optimizer._deep_merge(base, {"portfolio": {"crash_reinvestment": {"enabled": False}}})

    with_reinvest, _ = optimizer._simulate_strategy_fraction(1.0, universe, base)
    without_reinvest, _ = optimizer._simulate_strategy_fraction(1.0, universe, disabled)

    assert math.isclose(with_reinvest[0], without_reinvest[0], rel_tol=0.0, abs_tol=1e-9)


def test_crash_reinvestment_runs_once_per_path():
    universe = _custom_universe([-0.30, 0.40, -0.30, 0.40, 0.10], unemployed=False)
    cfg = optimizer._deep_merge(
        optimizer.DEFAULT_CONFIG,
        {
            "portfolio": {
                "initial_portfolio": 100.0,
                "initial_basis": 100.0,
                "ltcg_tax_rate": 0.0,
                "monthly_expenses": 0.0,
                "monthly_savings": 0.0,
                "crash_reinvestment": {
                    "enabled": True,
                    "crash_drawdown_threshold": 0.20,
                    "recovery_fraction_of_peak": 0.90,
                    "reinvest_fraction_of_initial_sale_proceeds": 0.5,
                    "cash_buffer_months": 0,
                },
            },
            "market": {"cash_yield_annual": 0.0},
        },
    )

    final_net_worth, ruined = optimizer._simulate_strategy_fraction(1.0, universe, cfg)
    assert not ruined[0]
    assert 103.0 < final_net_worth[0] < 105.0


def test_crash_reinvestment_works_with_retirement_enabled():
    universe = _custom_universe([-0.30, 0.20, 0.10, 0.10], unemployed=False)
    base = optimizer._deep_merge(
        optimizer.DEFAULT_CONFIG,
        {
            "portfolio": {
                "initial_portfolio": 100.0,
                "initial_basis": 100.0,
                "ltcg_tax_rate": 0.0,
                "monthly_expenses": 0.0,
                "monthly_savings": 10.0,
                "retirement": {
                    "enabled": True,
                    "start_month_from_start": 0,
                    "safe_withdrawal_rate_annual": 0.0,
                    "expense_reduction_fraction": 0.8,
                    "dynamic_safe_withdrawal_rate": False,
                },
                "crash_reinvestment": {
                    "enabled": True,
                    "crash_drawdown_threshold": 0.20,
                    "recovery_fraction_of_peak": 0.90,
                    "reinvest_fraction_of_initial_sale_proceeds": 1.0,
                    "cash_buffer_months": 0,
                },
            },
            "market": {"cash_yield_annual": 0.0},
        },
    )
    disabled = optimizer._deep_merge(base, {"portfolio": {"crash_reinvestment": {"enabled": False}}})

    with_reinvest, ruined_with = optimizer._simulate_strategy_fraction(1.0, universe, base)
    without_reinvest, ruined_without = optimizer._simulate_strategy_fraction(1.0, universe, disabled)

    assert not ruined_with[0]
    assert not ruined_without[0]
    assert with_reinvest[0] > without_reinvest[0] + 5.0


def test_run_result_includes_reinvestment_metadata():
    cfg = _base_config()
    cfg["portfolio"] = {
        "crash_reinvestment": {
            "enabled": True,
        }
    }
    result = optimizer.run_monte_carlo_optimization(config=cfg, verbose=False)
    assert result["reinvestment"]["enabled"] is True
    assert result["execution"]["reinvestment"]["frequency"] == "once_per_path"


@pytest.mark.parametrize("objective_mode", sorted(optimizer.SUPPORTED_OBJECTIVES))
def test_single_thread_smoke_runs_all_objectives(objective_mode):
    cfg = _base_config()
    cfg["risk"] = {"objective_mode": objective_mode}
    result = optimizer.run_monte_carlo_optimization(config=cfg, verbose=False)
    _assert_core_result_shape(result, expected_points=cfg["decision_grid"]["num_points"])
    assert result["execution"]["mode"] == "single"
    assert result["primary_objective_mode"] == objective_mode
    assert set(result["objective_comparison"]) == optimizer.SUPPORTED_OBJECTIVES


@pytest.mark.parametrize("objective_mode", sorted(optimizer.SUPPORTED_OBJECTIVES))
def test_parallel_smoke_runs_all_objectives(objective_mode):
    cfg = _base_config()
    cfg["simulation"]["parallel_enabled"] = True
    cfg["simulation"]["parallel_workers"] = 2
    cfg["risk"] = {"objective_mode": objective_mode}
    result = optimizer.run_monte_carlo_optimization(config=cfg, verbose=False)
    _assert_core_result_shape(result, expected_points=cfg["decision_grid"]["num_points"])
    assert result["execution"]["workers_used"] >= 1
    assert result["primary_objective_mode"] == objective_mode
    assert set(result["objective_comparison"]) == optimizer.SUPPORTED_OBJECTIVES


def test_single_vs_parallel_consistency_cvar():
    base = _base_config()
    single = optimizer.run_monte_carlo_optimization(config=base, verbose=False)

    parallel_cfg = optimizer._deep_merge(base, {"simulation": {"parallel_enabled": True, "parallel_workers": 2}})
    parallel = optimizer.run_monte_carlo_optimization(config=parallel_cfg, verbose=False)
    _assert_close_metrics(single, parallel)


def test_single_vs_parallel_consistency_log_utility():
    base = _base_config()
    base["risk"] = {"objective_mode": "expected_log_utility"}
    single = optimizer.run_monte_carlo_optimization(config=base, verbose=False)

    parallel_cfg = optimizer._deep_merge(base, {"simulation": {"parallel_enabled": True, "parallel_workers": 2}})
    parallel = optimizer.run_monte_carlo_optimization(config=parallel_cfg, verbose=False)
    _assert_close_metrics(single, parallel)


def test_single_vs_parallel_consistency_consensus():
    base = _base_config()
    base["risk"] = {"objective_mode": "consensus"}
    single = optimizer.run_monte_carlo_optimization(config=base, verbose=False)

    parallel_cfg = optimizer._deep_merge(base, {"simulation": {"parallel_enabled": True, "parallel_workers": 2}})
    parallel = optimizer.run_monte_carlo_optimization(config=parallel_cfg, verbose=False)
    _assert_close_metrics(single, parallel)

    consensus = parallel["objective_comparison"]["consensus"]
    assert "consensus_max_regret" in consensus
    assert "consensus_within_tolerance_count" in consensus
    assert "consensus_used_fallback" in consensus


def test_guardrail_behavior_identical_across_modes():
    stress = {
        "simulation": {
            "n_sims": 3_000,
            "seed": 7,
            "min_horizon_months": 24,
            "post_crash_buffer_months": 0,
            "parallel_enabled": False,
        },
        "gpu": {
            "enabled": False,
        },
        "portfolio": {
            "initial_portfolio": 20_000,
            "initial_basis": 10_000,
            "monthly_expenses": 8_000,
            "monthly_savings": 0,
        },
        "beliefs": {
            "prob_crash": 1.0,
            "crash_start_month": 1,
            "crash_duration_months": 24,
            "crash_severity_drop": 0.9,
            "prob_layoff_during_crash": 1.0,
            "prob_layoff_baseline": 1.0,
        },
        "risk": {
            "max_ruin_probability": 0.0,
            "objective_mode": "consensus",
        },
        "decision_grid": {"num_points": 11},
    }

    with pytest.raises(ValueError, match="No strategy satisfies the ruin probability guardrail"):
        optimizer.run_monte_carlo_optimization(config=stress, verbose=False)

    stress_parallel = optimizer._deep_merge(
        stress,
        {"simulation": {"parallel_enabled": True, "parallel_workers": 2}},
    )
    with pytest.raises(ValueError, match="No strategy satisfies the ruin probability guardrail"):
        optimizer.run_monte_carlo_optimization(config=stress_parallel, verbose=False)


def test_parallel_worker_resolution_uses_all_cores_default(monkeypatch):
    monkeypatch.setattr(optimizer.os, "cpu_count", lambda: 8)
    config = optimizer._deep_merge(
        optimizer.DEFAULT_CONFIG,
        {"simulation": {"parallel_enabled": True, "parallel_workers": None}},
    )
    settings = optimizer._resolve_execution_settings(config, task_count=3)
    assert settings["mode"] == "parallel"
    assert settings["workers_used"] == 3


def test_parallel_can_be_forced_single_worker():
    cfg = _base_config()
    cfg["simulation"]["parallel_enabled"] = True
    cfg["simulation"]["parallel_workers"] = 1
    result = optimizer.run_monte_carlo_optimization(config=cfg, verbose=False)
    assert result["execution"]["mode"] == "single"
    assert result["execution"]["workers_used"] == 1


def test_tail_reduction_worker_resolution_respects_explicit_override():
    config = optimizer._deep_merge(
        optimizer.DEFAULT_CONFIG,
        {"gpu": {"reduction_workers": 8}},
    )
    workers = optimizer._resolve_tail_reduction_workers(config, task_count=3, values_per_task=10_000_000)
    assert workers == 3


def test_tail_reduction_worker_resolution_caps_with_memory_budget(monkeypatch):
    monkeypatch.setattr(optimizer.os, "cpu_count", lambda: 16)
    monkeypatch.setattr(optimizer, "_get_available_memory_bytes", lambda: 4 * 1024 * 1024 * 1024)
    config = optimizer._deep_merge(
        optimizer.DEFAULT_CONFIG,
        {"gpu": {"reduction_workers": None}},
    )
    workers = optimizer._resolve_tail_reduction_workers(config, task_count=12, values_per_task=10_000_000)
    assert workers == 1


def test_tail_reduction_worker_resolution_uses_cpu_count_when_memory_allows(monkeypatch):
    monkeypatch.setattr(optimizer.os, "cpu_count", lambda: 16)
    monkeypatch.setattr(optimizer, "_get_available_memory_bytes", lambda: 32 * 1024 * 1024 * 1024)
    config = optimizer._deep_merge(
        optimizer.DEFAULT_CONFIG,
        {"gpu": {"reduction_workers": None}},
    )
    workers = optimizer._resolve_tail_reduction_workers(config, task_count=12, values_per_task=10_000_000)
    assert workers == 12


def test_tail_metric_helper_matches_reference():
    rng = np.random.default_rng(7)
    final_values = rng.normal(loc=2_000_000.0, scale=250_000.0, size=25_000).astype(np.float64)
    floor = 2_200_000.0
    alpha = 0.10

    expected_p10 = float(np.percentile(final_values, 10))
    expected_shortfall = np.maximum(0.0, floor - final_values)
    expected_cvar = optimizer._compute_cvar(expected_shortfall, alpha)

    p10, cvar = optimizer._compute_tail_metrics_from_final_values(final_values, floor, alpha)
    assert math.isclose(p10, expected_p10, rel_tol=1e-12, abs_tol=1e-10)
    assert math.isclose(cvar, expected_cvar, rel_tol=1e-12, abs_tol=1e-10)

    inplace_values = final_values.copy()
    p10_inplace, cvar_inplace = optimizer._compute_tail_metrics_from_final_values(
        inplace_values,
        floor,
        alpha,
        allow_inplace=True,
    )
    assert math.isclose(p10_inplace, expected_p10, rel_tol=1e-12, abs_tol=1e-10)
    assert math.isclose(cvar_inplace, expected_cvar, rel_tol=1e-12, abs_tol=1e-10)


def test_fraction_display_decimals_tracks_grid_step():
    fractions = np.linspace(0.20, 0.30, 1001)
    assert optimizer._fraction_display_decimals(fractions) == 2

    coarse = np.linspace(0.0, 1.0, 51)
    assert optimizer._fraction_display_decimals(coarse) == 1


def test_build_table_display_indices_samples_evenly():
    indices = optimizer._build_table_display_indices(1001, max_rows=121)
    assert indices[0] == 0
    assert indices[-1] == 1000
    assert len(indices) <= 121
    assert len(indices) > 2


def test_gpu_extension_fallback_to_cpu_when_unavailable(monkeypatch):
    def fake_load_failure():
        raise optimizer.gpu_backend.GpuBackendError("forced missing extension for test")

    monkeypatch.setattr(optimizer.gpu_backend, "load_cuda_extension", fake_load_failure)

    cfg = _base_config()
    cfg["gpu"] = {
        "enabled": True,
        "prefer_cuda_extension": True,
        "fallback_to_cpu_on_error": True,
    }
    result = optimizer.run_monte_carlo_optimization(config=cfg, verbose=False)
    assert result["execution"]["backend"] == "numpy_streaming"
    assert "forced missing extension" in result["execution"]["fallback_reason"]


def test_synthetic_generation_params_support_beta_prob_crash():
    cfg = optimizer._deep_merge(
        optimizer.DEFAULT_CONFIG,
        {
            "beliefs": {
                "prob_crash": {
                    "dist": "beta",
                    "a": 3.0,
                    "b": 4.0,
                    "low": 0.15,
                    "high": 0.75,
                }
            }
        },
    )

    params = optimizer._build_synthetic_generation_params(cfg, horizon_months=120, chunk_start=0)
    assert params is not None
    assert params["prob_crash_mode"] == 3
    assert math.isclose(params["prob_crash_a"], 3.0)
    assert math.isclose(params["prob_crash_b"], 4.0)
    assert math.isclose(params["prob_crash_c"], 0.15)
    assert math.isclose(params["prob_crash_d"], 0.75)


def test_can_use_device_scenario_generation_accepts_beta_prob_crash(monkeypatch):
    monkeypatch.setattr(optimizer.gpu_backend, "load_cuda_extension", lambda: object())
    monkeypatch.setattr(optimizer.gpu_backend, "supports_synthetic_generation", lambda ext: True)

    cfg = optimizer._deep_merge(
        optimizer.DEFAULT_CONFIG,
        {
            "beliefs": {
                "prob_crash": {
                    "dist": "beta",
                    "a": 2.0,
                    "b": 5.0,
                    "low": 0.05,
                    "high": 0.8,
                }
            }
        },
    )
    assert optimizer._can_use_device_scenario_generation(cfg, use_cuda_backend=True)


def test_synthetic_generation_params_reject_unsupported_prob_crash_distribution():
    cfg = optimizer._deep_merge(
        optimizer.DEFAULT_CONFIG,
        {
            "beliefs": {
                "prob_crash": {
                    "dist": "lognormal",
                    "mean": -3.0,
                    "sigma": 0.5,
                    "clip_low": 0.0,
                    "clip_high": 1.0,
                }
            }
        },
    )
    assert optimizer._build_synthetic_generation_params(cfg, horizon_months=120, chunk_start=0) is None


def test_numpy_streaming_backend_runs_when_cuda_not_required():
    cfg = _base_config()
    cfg["gpu"] = {
        "enabled": True,
        "prefer_cuda_extension": False,
        "scenario_chunk_size": 256,
        "sample_size": 500,
    }
    result = optimizer.run_monte_carlo_optimization(config=cfg, verbose=False)
    assert result["execution"]["backend"] == "numpy_streaming"
    assert result["execution"]["chunk_size_used"] == 256


def test_exact_two_pass_mode_runs_and_reports_mode():
    cfg = _base_config()
    cfg["gpu"] = {
        "enabled": True,
        "prefer_cuda_extension": False,
        "cvar_mode": "exact_two_pass",
        "scenario_chunk_size": 300,
    }
    result = optimizer.run_monte_carlo_optimization(config=cfg, verbose=False)
    assert result["execution"]["backend"] == "numpy_streaming"
    assert result["execution"]["cvar_mode"] == "exact_two_pass"
    assert math.isfinite(result["cvar_shortfall"])


def test_autotune_streaming_geometry_reduces_chunk_or_tile_when_memory_small():
    cfg = optimizer._deep_merge(
        optimizer.DEFAULT_CONFIG,
        {
            "simulation": {"n_sims": 50_000},
            "gpu": {
                "scenario_chunk_size": 50_000,
                "fraction_tile_size": 51,
                "max_vram_utilization": 0.5,
                "min_chunk_size": 2_000,
            },
        },
    )
    chunk_size, tile_size, note = optimizer._autotune_streaming_geometry(
        cfg,
        n_fractions=51,
        horizon_months=240,
        gpu_meta={"device_memory_total_mb": 128, "device_memory_free_mb": 64},
        use_cuda_backend=True,
    )
    assert chunk_size <= 50_000
    assert tile_size <= 51
    assert note is not None


def test_cuda_oom_backoff_retries_with_smaller_chunk(monkeypatch):
    calls = {"count": 0}

    monkeypatch.setattr(optimizer.gpu_backend, "load_cuda_extension", lambda: object())
    monkeypatch.setattr(
        optimizer.gpu_backend,
        "query_device_metadata",
        lambda device_id: {
            "gpu_name": "FakeGPU",
            "driver_version": "0",
            "device_memory_total_mb": 32000,
            "device_memory_free_mb": 32000,
        },
    )

    def fake_simulate_fraction_tile(monthly_returns, unemployment_matrix, fractions, portfolio_config, market_config):
        calls["count"] += 1
        if calls["count"] == 1:
            raise optimizer.gpu_backend.GpuBackendError("cudaErrorMemoryAllocation: out of memory")
        tile_n = len(fractions)
        n = monthly_returns.shape[0]
        final = np.full((tile_n, n), 1_000_000.0, dtype=np.float64)
        ruined = np.zeros((tile_n, n), dtype=bool)
        return final, ruined, 0.1

    monkeypatch.setattr(optimizer.gpu_backend, "simulate_fraction_tile", fake_simulate_fraction_tile)

    cfg = _base_config()
    cfg["simulation"]["n_sims"] = 3_000
    cfg["gpu"] = {
        "enabled": True,
        "prefer_cuda_extension": True,
        "scenario_chunk_size": 1_000,
        "fraction_tile_size": 5,
        "min_chunk_size": 200,
        "oom_backoff_factor": 0.5,
        "cvar_refine_pass": False,
    }
    result = optimizer.run_monte_carlo_optimization(config=cfg, verbose=False)
    assert result["execution"]["backend"] == "cuda"
    assert result["execution"]["chunk_size_used"] < 1_000
    assert "backoff" in result["execution"]["fallback_reason"].lower()
    assert calls["count"] >= 2


def test_refine_pass_skips_exact_when_resource_guard_blocks(monkeypatch):
    monkeypatch.setattr(
        optimizer,
        "_can_run_exact_tail_pass",
        lambda config, subset_size, values_per_task: (False, "resource guard triggered for test"),
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("_compute_exact_tail_metrics_via_memmap should not be called when guard blocks refine")

    monkeypatch.setattr(optimizer, "_compute_exact_tail_metrics_via_memmap", fail_if_called)

    cfg = _base_config()
    cfg["gpu"] = {
        "enabled": True,
        "prefer_cuda_extension": False,
        "scenario_chunk_size": 300,
        "sample_size": 500,
        "cvar_mode": "streaming_near_exact",
        "cvar_refine_pass": True,
        "refine_top_k": 3,
    }
    result = optimizer.run_monte_carlo_optimization(config=cfg, verbose=False)
    assert "resource guard triggered for test" in result["execution"]["fallback_reason"]


def test_exact_two_pass_fails_fast_when_resource_guard_blocks(monkeypatch):
    monkeypatch.setattr(
        optimizer,
        "_can_run_exact_tail_pass",
        lambda config, subset_size, values_per_task: (False, "insufficient resources for test"),
    )

    cfg = _base_config()
    cfg["gpu"] = {
        "enabled": True,
        "prefer_cuda_extension": False,
        "cvar_mode": "exact_two_pass",
        "scenario_chunk_size": 300,
    }
    with pytest.raises(RuntimeError, match="insufficient resources for test"):
        optimizer.run_monte_carlo_optimization(config=cfg, verbose=False)


def test_progress_callback_emits_core_lifecycle_events():
    cfg = _base_config()
    cfg["decision_grid"]["num_points"] = 7

    events = []

    def progress_callback(event, payload):
        events.append((event, payload))

    result = optimizer.run_monte_carlo_optimization(
        config=cfg,
        verbose=False,
        progress_callback=progress_callback,
    )

    _assert_core_result_shape(result, expected_points=cfg["decision_grid"]["num_points"])

    event_names = [name for name, _ in events]
    assert "run_start" in event_names
    assert "run_complete" in event_names
    assert "cpu_fraction_eval_start" in event_names
    assert "cpu_fraction_eval_complete" in event_names
    assert any(name == "cpu_fraction_eval_progress" for name in event_names)
