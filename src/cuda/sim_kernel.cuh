#pragma once

#include <cstdint>
#include <cuda_runtime.h>

void launch_simulate_fraction_tile(
    const float* monthly_returns,
    const std::uint8_t* unemployment_matrix,
    const double* fractions,
    int n_sims,
    int horizon,
    int n_fractions,
    double initial_portfolio,
    double initial_basis,
    double ltcg_tax_rate,
    double monthly_expenses,
    double monthly_savings,
    double cash_yield_annual,
    double* out_final_net_worth,
    std::uint8_t* out_ruined
);

void launch_simulate_fraction_tile_aggregates(
    const float* monthly_returns,
    const std::uint8_t* unemployment_matrix,
    const double* fractions,
    int n_sims,
    int horizon,
    int fraction_start,
    int fraction_count,
    double initial_portfolio,
    double initial_basis,
    double ltcg_tax_rate,
    double monthly_expenses,
    double monthly_savings,
    double cash_yield_annual,
    double log_utility_wealth_floor,
    const int* sample_map,
    int sample_count,
    double* out_sum_final,
    double* out_sum_log,
    unsigned long long* out_ruin_count,
    double* out_sampled_final,
    cudaStream_t stream
);
