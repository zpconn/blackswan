import importlib
import subprocess
import time

import numpy as np


class GpuBackendError(RuntimeError):
    pass


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
        return importlib.import_module("crash_prep_cuda")
    except Exception as exc:
        raise GpuBackendError(
            "CUDA extension 'crash_prep_cuda' is unavailable. "
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
