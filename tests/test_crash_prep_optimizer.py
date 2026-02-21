import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import crash_prep_optimizer as optimizer


def _base_config():
    return {
        "simulation": {
            "n_sims": 1_500,
            "seed": 123,
            "parallel_enabled": False,
        },
        "decision_grid": {
            "num_points": 9,
        },
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
    config = optimizer._deep_merge(optimizer.DEFAULT_CONFIG, {"simulation": {"parallel_workers": None}})
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
