#include "sim_kernel.cuh"

#include <cmath>
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace {

__device__ inline void simulate_one_path(
    const float* monthly_returns,
    const std::uint8_t* unemployment_matrix,
    int horizon,
    int scenario_idx,
    double fraction_sold,
    double initial_portfolio,
    double initial_basis,
    double ltcg_tax_rate,
    double monthly_expenses,
    double monthly_savings,
    double cash_yield_annual,
    double* final_net_worth_out,
    bool* ruined_out
) {
    double gross_sold = fraction_sold * initial_portfolio;
    double basis_sold = fraction_sold * initial_basis;
    double capital_gains = fmax(0.0, gross_sold - basis_sold);
    double tax_paid = capital_gains * ltcg_tax_rate;

    double cash = gross_sold - tax_paid;
    double stocks = (1.0 - fraction_sold) * initial_portfolio;
    double basis = (1.0 - fraction_sold) * initial_basis;
    double market_index = 1.0;
    bool ruined = false;
    double r_cash = cash_yield_annual / 12.0;

    for (int month_idx = 0; month_idx < horizon; ++month_idx) {
        int offset = scenario_idx * horizon + month_idx;
        double monthly_return = static_cast<double>(monthly_returns[offset]);
        market_index *= (1.0 + monthly_return);
        cash *= (1.0 + r_cash);

        bool is_unemployed = unemployment_matrix[offset] != 0;
        if (!is_unemployed) {
            stocks += monthly_savings / market_index;
            basis += monthly_savings;
        }

        double need = is_unemployed ? monthly_expenses : 0.0;
        double from_cash = fmin(need, cash);
        cash -= from_cash;
        need -= from_cash;

        if (need > 1e-9) {
            if (stocks <= 1e-12) {
                ruined = true;
            } else {
                double basis_per_unit = basis / stocks;
                double gain_per_unit = fmax(0.0, market_index - basis_per_unit);
                double tax_per_unit = gain_per_unit * ltcg_tax_rate;
                double net_per_unit = fmax(1e-12, market_index - tax_per_unit);
                double units_to_sell = need / net_per_unit;
                if (units_to_sell > stocks) {
                    units_to_sell = stocks;
                }

                double proceeds = units_to_sell * net_per_unit;
                double remaining_need = need - proceeds;
                if (remaining_need > 1e-9) {
                    ruined = true;
                }

                double prev_stocks = stocks;
                double s_new = prev_stocks - units_to_sell;
                double b_new = basis * (s_new / prev_stocks);
                stocks = s_new;
                basis = b_new;
            }
        }
    }

    double portfolio_value = stocks * market_index;
    double final_gains = fmax(0.0, portfolio_value - basis);
    double final_tax = final_gains * ltcg_tax_rate;
    double final_net_worth = cash + portfolio_value - final_tax;
    if (final_net_worth <= 0.0) {
        ruined = true;
    }

    *final_net_worth_out = final_net_worth;
    *ruined_out = ruined;
}

__global__ void simulate_fraction_tile_kernel(
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
) {
    int linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_sims * n_fractions;
    if (linear_idx >= total) {
        return;
    }

    int scenario_idx = linear_idx % n_sims;
    int fraction_idx = linear_idx / n_sims;

    double final_net_worth = 0.0;
    bool ruined = false;
    simulate_one_path(
        monthly_returns,
        unemployment_matrix,
        horizon,
        scenario_idx,
        fractions[fraction_idx],
        initial_portfolio,
        initial_basis,
        ltcg_tax_rate,
        monthly_expenses,
        monthly_savings,
        cash_yield_annual,
        &final_net_worth,
        &ruined
    );

    int out_idx = fraction_idx * n_sims + scenario_idx;
    out_final_net_worth[out_idx] = final_net_worth;
    out_ruined[out_idx] = ruined ? 1 : 0;
}

__global__ void simulate_fraction_tile_reduced_kernel(
    const float* monthly_returns,
    const std::uint8_t* unemployment_matrix,
    const double* fractions,
    int n_sims,
    int horizon,
    int fraction_start,
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
    double* out_sampled_final
) {
    extern __shared__ double shared[];
    double* shared_sum_final = shared;
    double* shared_sum_log = shared + blockDim.x;
    unsigned int* shared_ruin = reinterpret_cast<unsigned int*>(shared_sum_log + blockDim.x);

    int local_fraction_idx = blockIdx.y;
    int fraction_idx = fraction_start + local_fraction_idx;
    int scenario_idx = blockIdx.x * blockDim.x + threadIdx.x;

    double final_net_worth = 0.0;
    double log_score = 0.0;
    unsigned int ruined = 0;

    if (scenario_idx < n_sims) {
        bool scenario_ruined = false;
        simulate_one_path(
            monthly_returns,
            unemployment_matrix,
            horizon,
            scenario_idx,
            fractions[fraction_idx],
            initial_portfolio,
            initial_basis,
            ltcg_tax_rate,
            monthly_expenses,
            monthly_savings,
            cash_yield_annual,
            &final_net_worth,
            &scenario_ruined
        );
        log_score = log(fmax(final_net_worth, log_utility_wealth_floor));
        ruined = scenario_ruined ? 1u : 0u;

        if (sample_map != nullptr && out_sampled_final != nullptr && sample_count > 0) {
            int sample_idx = sample_map[scenario_idx];
            if (sample_idx >= 0 && sample_idx < sample_count) {
                std::size_t sample_offset = static_cast<std::size_t>(fraction_idx) * sample_count + sample_idx;
                out_sampled_final[sample_offset] = final_net_worth;
            }
        }
    }

    shared_sum_final[threadIdx.x] = final_net_worth;
    shared_sum_log[threadIdx.x] = log_score;
    shared_ruin[threadIdx.x] = ruined;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            shared_sum_final[threadIdx.x] += shared_sum_final[threadIdx.x + stride];
            shared_sum_log[threadIdx.x] += shared_sum_log[threadIdx.x + stride];
            shared_ruin[threadIdx.x] += shared_ruin[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(&out_sum_final[fraction_idx], shared_sum_final[0]);
        atomicAdd(&out_sum_log[fraction_idx], shared_sum_log[0]);
        atomicAdd(&out_ruin_count[fraction_idx], static_cast<unsigned long long>(shared_ruin[0]));
    }
}

}  // namespace

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
) {
    int total = n_sims * n_fractions;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    simulate_fraction_tile_kernel<<<blocks, threads>>>(
        monthly_returns,
        unemployment_matrix,
        fractions,
        n_sims,
        horizon,
        n_fractions,
        initial_portfolio,
        initial_basis,
        ltcg_tax_rate,
        monthly_expenses,
        monthly_savings,
        cash_yield_annual,
        out_final_net_worth,
        out_ruined
    );
}

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
) {
    if (fraction_count <= 0 || n_sims <= 0 || horizon <= 0) {
        return;
    }

    int threads = 256;
    int blocks_x = (n_sims + threads - 1) / threads;
    dim3 grid(static_cast<unsigned int>(blocks_x), static_cast<unsigned int>(fraction_count), 1u);
    std::size_t shared_bytes = static_cast<std::size_t>(threads) * (2 * sizeof(double) + sizeof(unsigned int));

    simulate_fraction_tile_reduced_kernel<<<grid, threads, shared_bytes, stream>>>(
        monthly_returns,
        unemployment_matrix,
        fractions,
        n_sims,
        horizon,
        fraction_start,
        initial_portfolio,
        initial_basis,
        ltcg_tax_rate,
        monthly_expenses,
        monthly_savings,
        cash_yield_annual,
        log_utility_wealth_floor,
        sample_map,
        sample_count,
        out_sum_final,
        out_sum_log,
        out_ruin_count,
        out_sampled_final
    );
}
