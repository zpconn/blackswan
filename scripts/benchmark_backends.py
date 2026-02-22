#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blackswan import run_monte_carlo_optimization


def run_case(name, config):
    t0 = time.perf_counter()
    try:
        result = run_monte_carlo_optimization(config=config, verbose=False)
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        print(f"\n{name}")
        print(f"  elapsed_s:             {elapsed:.3f}")
        print(f"  status:                FAILED")
        print(f"  error:                 {exc!r}")
        return None, None
    elapsed = time.perf_counter() - t0
    execution = result["execution"]
    print(f"\n{name}")
    print(f"  elapsed_s:             {elapsed:.3f}")
    print(f"  backend:               {execution['backend']}")
    print(f"  chunk_size_used:       {execution.get('chunk_size_used')}")
    print(f"  fraction_tile_used:    {execution.get('fraction_tile_used')}")
    print(f"  kernel_time_ms:        {execution.get('kernel_time_ms')}")
    print(f"  transfer_time_ms:      {execution.get('transfer_time_ms')}")
    print(f"  reduction_time_ms:     {execution.get('reduction_time_ms')}")
    print(f"  fallback_reason:       {execution.get('fallback_reason')}")
    print(f"  recommended_fraction:  {result['recommended_fraction']:.4f}")
    print(f"  expected_final_wealth: {result['expected_final_net_worth']:.2f}")
    print(f"  cvar_shortfall:        {result['cvar_shortfall']:.2f}")
    return elapsed, result


def main():
    parser = argparse.ArgumentParser(description="Benchmark CUDA vs numpy streaming backends.")
    parser.add_argument("--n-sims", type=int, default=200_000)
    parser.add_argument("--num-points", type=int, default=21)
    parser.add_argument("--chunk-size", type=int, default=50_000)
    parser.add_argument("--sample-size", type=int, default=50_000)
    args = parser.parse_args()

    base = {
        "simulation": {
            "n_sims": args.n_sims,
            "seed": 123,
            "parallel_enabled": False,
        },
        "decision_grid": {
            "num_points": args.num_points,
        },
        "risk": {
            "objective_mode": "consensus",
        },
    }

    cuda_cfg = {
        **base,
        "gpu": {
            "enabled": True,
            "prefer_cuda_extension": True,
            "fallback_to_cpu_on_error": False,
            "scenario_chunk_size": args.chunk_size,
            "sample_size": args.sample_size,
            "cvar_mode": "streaming_near_exact",
            "cvar_refine_pass": True,
            "refine_top_k": 5,
        },
    }
    stream_cfg = {
        **base,
        "gpu": {
            "enabled": True,
            "prefer_cuda_extension": False,
            "scenario_chunk_size": args.chunk_size,
            "sample_size": args.sample_size,
            "cvar_mode": "streaming_near_exact",
            "cvar_refine_pass": True,
            "refine_top_k": 5,
        },
    }

    cuda_elapsed, _ = run_case("CUDA Extension Path", cuda_cfg)
    stream_elapsed, _ = run_case("NumPy Streaming Path", stream_cfg)

    if cuda_elapsed is not None and cuda_elapsed > 0 and stream_elapsed is not None:
        print(f"\nSpeedup (stream/cuda): {stream_elapsed / cuda_elapsed:.2f}x")


if __name__ == "__main__":
    main()
