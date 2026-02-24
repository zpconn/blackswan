import importlib
import subprocess
import time

import numpy as np


class GpuBackendError(RuntimeError):
    pass


def _retirement_kernel_args(portfolio_config):
    retirement = (portfolio_config or {}).get("retirement") or {}
    enabled = bool(retirement.get("enabled", False))
    start_month = retirement.get("start_month_from_start")
    if start_month is None:
        start_month = -1

    return (
        int(enabled),
        int(start_month),
        float(retirement.get("safe_withdrawal_rate_annual", 0.04)),
        float(retirement.get("expense_reduction_fraction", 1.0)),
        int(bool(retirement.get("dynamic_safe_withdrawal_rate", False))),
    )


def _reinvestment_kernel_args(portfolio_config):
    reinvest = (portfolio_config or {}).get("crash_reinvestment") or {}
    return (
        int(bool(reinvest.get("enabled", False))),
        float(reinvest.get("crash_drawdown_threshold", 0.20)),
        float(reinvest.get("recovery_fraction_of_peak", 0.90)),
        float(reinvest.get("reinvest_fraction_of_initial_sale_proceeds", 1.0)),
        int(reinvest.get("cash_buffer_months", 12)),
    )


def _parse_nvidia_smi_line(line):
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != 4:
        return {
            "gpu_name": None,
            "driver_version": None,
            "device_memory_total_mb": None,
            "device_memory_free_mb": None,
        }
    try:
        memory_total_mb = int(parts[2])
    except ValueError:
        memory_total_mb = None
    try:
        memory_free_mb = int(parts[3])
    except ValueError:
        memory_free_mb = None
    return {
        "gpu_name": parts[0],
        "driver_version": parts[1],
        "device_memory_total_mb": memory_total_mb,
        "device_memory_free_mb": memory_free_mb,
    }


def query_device_metadata(device_id):
    try:
        cmd = [
            "nvidia-smi",
            f"--id={int(device_id)}",
            "--query-gpu=name,driver_version,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        first_line = proc.stdout.strip().splitlines()[0]
    except Exception:
        return {
            "gpu_name": None,
            "driver_version": None,
            "device_memory_total_mb": None,
            "device_memory_free_mb": None,
        }
    return _parse_nvidia_smi_line(first_line)


def load_cuda_extension():
    try:
        return importlib.import_module("blackswan_cuda")
    except Exception as exc:
        raise GpuBackendError(
            "CUDA extension 'blackswan_cuda' is unavailable. "
            "Build it first with the provided CMake/pyproject config. "
            f"Original import error: {exc!r}"
        ) from exc


def _simulate_fraction_tile_legacy(
    ext,
    monthly_returns,
    unemployment_matrix,
    fractions,
    portfolio_config,
    market_config,
):
    (
        retirement_enabled,
        retirement_start_month_from_start,
        retirement_safe_withdrawal_rate_annual,
        retirement_expense_reduction_fraction,
        retirement_dynamic_safe_withdrawal_rate,
    ) = _retirement_kernel_args(portfolio_config)
    (
        reinvest_enabled,
        reinvest_crash_drawdown_threshold,
        reinvest_recovery_fraction_of_peak,
        reinvest_fraction_of_initial_sale_proceeds,
        reinvest_cash_buffer_months,
    ) = _reinvestment_kernel_args(portfolio_config)
    t0 = time.perf_counter()
    try:
        final_net_worth, ruined = ext.simulate_fraction_tile(
            np.ascontiguousarray(monthly_returns, dtype=np.float32),
            np.ascontiguousarray(unemployment_matrix, dtype=np.uint8),
            np.ascontiguousarray(fractions, dtype=np.float64),
            float(portfolio_config["initial_portfolio"]),
            float(portfolio_config["initial_basis"]),
            float(portfolio_config["ltcg_tax_rate"]),
            float(portfolio_config["monthly_expenses"]),
            float(portfolio_config["monthly_savings"]),
            retirement_enabled,
            retirement_start_month_from_start,
            retirement_safe_withdrawal_rate_annual,
            retirement_expense_reduction_fraction,
            retirement_dynamic_safe_withdrawal_rate,
            reinvest_enabled,
            reinvest_crash_drawdown_threshold,
            reinvest_recovery_fraction_of_peak,
            reinvest_fraction_of_initial_sale_proceeds,
            reinvest_cash_buffer_months,
            float(market_config["cash_yield_annual"]),
        )
    except Exception as exc:
        raise GpuBackendError(f"CUDA kernel execution failed: {exc}") from exc

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return final_net_worth, ruined.astype(bool), elapsed_ms


def simulate_fraction_tile(
    monthly_returns,
    unemployment_matrix,
    fractions,
    portfolio_config,
    market_config,
):
    ext = load_cuda_extension()
    return _simulate_fraction_tile_legacy(
        ext,
        monthly_returns,
        unemployment_matrix,
        fractions,
        portfolio_config,
        market_config,
    )


def simulate_fraction_tile_aggregates(
    monthly_returns,
    unemployment_matrix,
    fractions,
    portfolio_config,
    market_config,
    log_utility_wealth_floor,
    sample_positions=None,
    streams=1,
):
    ext = load_cuda_extension()
    monthly_returns = np.ascontiguousarray(monthly_returns, dtype=np.float32)
    unemployment_matrix = np.ascontiguousarray(unemployment_matrix, dtype=np.uint8)
    fractions = np.ascontiguousarray(fractions, dtype=np.float64)
    (
        retirement_enabled,
        retirement_start_month_from_start,
        retirement_safe_withdrawal_rate_annual,
        retirement_expense_reduction_fraction,
        retirement_dynamic_safe_withdrawal_rate,
    ) = _retirement_kernel_args(portfolio_config)
    (
        reinvest_enabled,
        reinvest_crash_drawdown_threshold,
        reinvest_recovery_fraction_of_peak,
        reinvest_fraction_of_initial_sale_proceeds,
        reinvest_cash_buffer_months,
    ) = _reinvestment_kernel_args(portfolio_config)

    if sample_positions is None:
        sample_positions_arr = None
    else:
        sample_positions_arr = np.ascontiguousarray(sample_positions, dtype=np.int64)

    if hasattr(ext, "simulate_fraction_tile_aggregates"):
        try:
            (
                sum_final,
                sum_log,
                ruin_count,
                sampled_final,
                kernel_ms,
                transfer_ms,
            ) = ext.simulate_fraction_tile_aggregates(
                monthly_returns,
                unemployment_matrix,
                fractions,
                float(portfolio_config["initial_portfolio"]),
                float(portfolio_config["initial_basis"]),
                float(portfolio_config["ltcg_tax_rate"]),
                float(portfolio_config["monthly_expenses"]),
                float(portfolio_config["monthly_savings"]),
                retirement_enabled,
                retirement_start_month_from_start,
                retirement_safe_withdrawal_rate_annual,
                retirement_expense_reduction_fraction,
                retirement_dynamic_safe_withdrawal_rate,
                reinvest_enabled,
                reinvest_crash_drawdown_threshold,
                reinvest_recovery_fraction_of_peak,
                reinvest_fraction_of_initial_sale_proceeds,
                reinvest_cash_buffer_months,
                float(market_config["cash_yield_annual"]),
                float(log_utility_wealth_floor),
                sample_positions_arr,
                int(streams),
            )
        except Exception as exc:
            raise GpuBackendError(f"CUDA aggregate kernel execution failed: {exc}") from exc
        return (
            np.asarray(sum_final, dtype=np.float64),
            np.asarray(sum_log, dtype=np.float64),
            np.asarray(ruin_count, dtype=np.int64),
            np.asarray(sampled_final, dtype=np.float64),
            float(kernel_ms),
            float(transfer_ms),
        )

    # Legacy compatibility path: fall back to full output and reduce on CPU.
    final_net_worth, ruined, kernel_ms = simulate_fraction_tile(
        monthly_returns,
        unemployment_matrix,
        fractions,
        portfolio_config,
        market_config,
    )
    sum_final = np.sum(final_net_worth, axis=1, dtype=np.float64)
    sum_log = np.sum(np.log(np.maximum(final_net_worth, log_utility_wealth_floor)), axis=1, dtype=np.float64)
    ruin_count = np.sum(ruined, axis=1, dtype=np.int64)
    if sample_positions_arr is None or sample_positions_arr.size == 0:
        sampled_final = np.empty((final_net_worth.shape[0], 0), dtype=np.float64)
    else:
        sampled_final = np.ascontiguousarray(final_net_worth[:, sample_positions_arr], dtype=np.float64)
    return sum_final, sum_log, ruin_count, sampled_final, float(kernel_ms), 0.0


def supports_synthetic_generation(ext=None):
    if ext is None:
        ext = load_cuda_extension()
    return hasattr(ext, "simulate_fraction_tile_aggregates_synthetic") and hasattr(
        ext, "simulate_fraction_tile_synthetic"
    )


def simulate_fraction_tile_aggregates_synthetic(
    n_sims,
    fractions,
    portfolio_config,
    market_config,
    log_utility_wealth_floor,
    synthetic_params,
    sample_positions=None,
    streams=1,
):
    ext = load_cuda_extension()
    if not hasattr(ext, "simulate_fraction_tile_aggregates_synthetic"):
        raise GpuBackendError("CUDA extension does not expose simulate_fraction_tile_aggregates_synthetic.")
    (
        retirement_enabled,
        retirement_start_month_from_start,
        retirement_safe_withdrawal_rate_annual,
        retirement_expense_reduction_fraction,
        retirement_dynamic_safe_withdrawal_rate,
    ) = _retirement_kernel_args(portfolio_config)
    (
        reinvest_enabled,
        reinvest_crash_drawdown_threshold,
        reinvest_recovery_fraction_of_peak,
        reinvest_fraction_of_initial_sale_proceeds,
        reinvest_cash_buffer_months,
    ) = _reinvestment_kernel_args(portfolio_config)

    fractions = np.ascontiguousarray(fractions, dtype=np.float64)
    if sample_positions is None:
        sample_positions_arr = None
    else:
        sample_positions_arr = np.ascontiguousarray(sample_positions, dtype=np.int64)

    try:
        (
            sum_final,
            sum_log,
            ruin_count,
            sampled_final,
            summary,
            kernel_ms,
            transfer_ms,
        ) = ext.simulate_fraction_tile_aggregates_synthetic(
            int(n_sims),
            fractions,
            float(portfolio_config["initial_portfolio"]),
            float(portfolio_config["initial_basis"]),
            float(portfolio_config["ltcg_tax_rate"]),
            float(portfolio_config["monthly_expenses"]),
            float(portfolio_config["monthly_savings"]),
            retirement_enabled,
            retirement_start_month_from_start,
            retirement_safe_withdrawal_rate_annual,
            retirement_expense_reduction_fraction,
            retirement_dynamic_safe_withdrawal_rate,
            reinvest_enabled,
            reinvest_crash_drawdown_threshold,
            reinvest_recovery_fraction_of_peak,
            reinvest_fraction_of_initial_sale_proceeds,
            reinvest_cash_buffer_months,
            float(market_config["cash_yield_annual"]),
            float(log_utility_wealth_floor),
            synthetic_params,
            sample_positions_arr,
            int(streams),
        )
    except Exception as exc:
        raise GpuBackendError(f"CUDA synthetic aggregate kernel execution failed: {exc}") from exc

    return (
        np.asarray(sum_final, dtype=np.float64),
        np.asarray(sum_log, dtype=np.float64),
        np.asarray(ruin_count, dtype=np.int64),
        np.asarray(sampled_final, dtype=np.float64),
        dict(summary),
        float(kernel_ms),
        float(transfer_ms),
    )


def simulate_fraction_tile_synthetic(
    n_sims,
    fractions,
    portfolio_config,
    market_config,
    synthetic_params,
    streams=1,
):
    ext = load_cuda_extension()
    if not hasattr(ext, "simulate_fraction_tile_synthetic"):
        raise GpuBackendError("CUDA extension does not expose simulate_fraction_tile_synthetic.")
    (
        retirement_enabled,
        retirement_start_month_from_start,
        retirement_safe_withdrawal_rate_annual,
        retirement_expense_reduction_fraction,
        retirement_dynamic_safe_withdrawal_rate,
    ) = _retirement_kernel_args(portfolio_config)
    (
        reinvest_enabled,
        reinvest_crash_drawdown_threshold,
        reinvest_recovery_fraction_of_peak,
        reinvest_fraction_of_initial_sale_proceeds,
        reinvest_cash_buffer_months,
    ) = _reinvestment_kernel_args(portfolio_config)

    fractions = np.ascontiguousarray(fractions, dtype=np.float64)
    try:
        final_net_worth, ruined, kernel_ms, transfer_ms = ext.simulate_fraction_tile_synthetic(
            int(n_sims),
            fractions,
            float(portfolio_config["initial_portfolio"]),
            float(portfolio_config["initial_basis"]),
            float(portfolio_config["ltcg_tax_rate"]),
            float(portfolio_config["monthly_expenses"]),
            float(portfolio_config["monthly_savings"]),
            retirement_enabled,
            retirement_start_month_from_start,
            retirement_safe_withdrawal_rate_annual,
            retirement_expense_reduction_fraction,
            retirement_dynamic_safe_withdrawal_rate,
            reinvest_enabled,
            reinvest_crash_drawdown_threshold,
            reinvest_recovery_fraction_of_peak,
            reinvest_fraction_of_initial_sale_proceeds,
            reinvest_cash_buffer_months,
            float(market_config["cash_yield_annual"]),
            synthetic_params,
            int(streams),
        )
    except Exception as exc:
        raise GpuBackendError(f"CUDA synthetic full kernel execution failed: {exc}") from exc

    return (
        np.asarray(final_net_worth, dtype=np.float64),
        np.asarray(ruined, dtype=bool),
        float(kernel_ms),
        float(transfer_ms),
    )
