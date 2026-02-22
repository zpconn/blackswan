#pragma once

#include <cstdint>
#include <cuda_runtime.h>

struct SyntheticGenerationParams {
    std::uint64_t seed;
    std::uint64_t global_offset;
    int horizon;
    double sigma;
    double monthly_market_drift;

    double prob_crash;

    int crash_start_mode;
    double crash_start_a;
    double crash_start_b;
    double crash_start_c;

    int crash_duration_mode;
    double crash_duration_a;
    double crash_duration_b;
    double crash_duration_c;

    int crash_severity_mode;
    double crash_severity_a;
    double crash_severity_b;
    double crash_severity_c;

    double prob_layoff_during_crash;
    double prob_layoff_baseline;
    int layoff_probabilities_are_annual;
    int crash_layoff_min_duration;
    int baseline_layoff_min_duration;
    int baseline_layoff_max_duration;
};

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

void launch_generate_scenario_params(
    int n_sims,
    const SyntheticGenerationParams& params,
    std::uint8_t* out_is_crash,
    int* out_crash_start,
    int* out_down_end,
    int* out_crash_end,
    double* out_mu_down,
    double* out_mu_up,
    int* out_layoff_start,
    int* out_layoff_end,
    unsigned long long* out_summary_counts,
    double* out_summary_sums,
    cudaStream_t stream
);

void launch_simulate_fraction_tile_aggregates_from_params(
    const std::uint8_t* is_crash,
    const int* crash_start,
    const int* down_end,
    const int* crash_end,
    const double* mu_down,
    const double* mu_up,
    const int* layoff_start,
    const int* layoff_end,
    const double* fractions,
    int n_sims,
    int fraction_start,
    int fraction_count,
    const SyntheticGenerationParams& params,
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

void launch_simulate_fraction_tile_full_from_params(
    const std::uint8_t* is_crash,
    const int* crash_start,
    const int* down_end,
    const int* crash_end,
    const double* mu_down,
    const double* mu_up,
    const int* layoff_start,
    const int* layoff_end,
    const double* fractions,
    int n_sims,
    int n_fractions,
    const SyntheticGenerationParams& params,
    double initial_portfolio,
    double initial_basis,
    double ltcg_tax_rate,
    double monthly_expenses,
    double monthly_savings,
    double cash_yield_annual,
    double* out_final_net_worth,
    std::uint8_t* out_ruined,
    cudaStream_t stream
);
