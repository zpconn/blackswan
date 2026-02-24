#include "sim_kernel.cuh"

#include <cmath>
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace {

constexpr std::uint64_t kMixA = 0x9e3779b97f4a7c15ULL;
constexpr std::uint64_t kMixB = 0xbf58476d1ce4e5b9ULL;
constexpr std::uint64_t kMixC = 0x94d049bb133111ebULL;
constexpr double kTwoPi = 6.283185307179586476925286766559;

__device__ inline std::uint64_t splitmix64(std::uint64_t x) {
    x += kMixA;
    x = (x ^ (x >> 30)) * kMixB;
    x = (x ^ (x >> 27)) * kMixC;
    x = x ^ (x >> 31);
    return x;
}

__device__ inline double u01_from_u64(std::uint64_t x) {
    return static_cast<double>((x >> 11) * (1.0 / 9007199254740992.0));
}

__device__ inline double random_u01(std::uint64_t seed, std::uint64_t scenario_id, std::uint64_t stream_id) {
    std::uint64_t x = seed;
    x ^= splitmix64(scenario_id + kMixA);
    x ^= splitmix64(stream_id + kMixB);
    return u01_from_u64(splitmix64(x));
}

__device__ inline double random_normal(std::uint64_t seed, std::uint64_t scenario_id, int month_idx) {
    std::uint64_t base = static_cast<std::uint64_t>(month_idx) * 2ULL;
    double u1 = fmax(random_u01(seed, scenario_id, 100000ULL + base), 1e-12);
    double u2 = random_u01(seed, scenario_id, 100001ULL + base);
    return sqrt(-2.0 * log(u1)) * cos(kTwoPi * u2);
}

__device__ inline double random_u01_counter(
    std::uint64_t seed,
    std::uint64_t scenario_id,
    std::uint64_t stream_base,
    std::uint64_t* counter
) {
    double value = random_u01(seed, scenario_id, stream_base + *counter);
    *counter += 1ULL;
    return value;
}

__device__ inline double random_normal_counter(
    std::uint64_t seed,
    std::uint64_t scenario_id,
    std::uint64_t stream_base,
    std::uint64_t* counter
) {
    double u1 = fmax(random_u01_counter(seed, scenario_id, stream_base, counter), 1e-12);
    double u2 = random_u01_counter(seed, scenario_id, stream_base, counter);
    return sqrt(-2.0 * log(u1)) * cos(kTwoPi * u2);
}

__device__ inline double clamp_probability(double p) {
    return fmin(1.0, fmax(0.0, p));
}

__device__ inline double sample_continuous(int mode, double a, double b, double c, double u) {
    if (mode == 0) {
        return a;
    }
    if (mode == 1) {
        return a + u * (b - a);
    }
    // triangular mode: a=left, b=mode, c=right
    double left = a;
    double peak = b;
    double right = c;
    if (!(left < right)) {
        return left;
    }
    double denom = right - left;
    double frac = (peak - left) / denom;
    frac = fmin(1.0, fmax(0.0, frac));
    if (u < frac) {
        return left + sqrt(fmax(0.0, u * denom * (peak - left)));
    }
    return right - sqrt(fmax(0.0, (1.0 - u) * denom * (right - peak)));
}

__device__ inline int sample_positive_int(int mode, double a, double b, double c, double u, int minimum) {
    double sampled = sample_continuous(mode, a, b, c, u);
    int v = static_cast<int>(llround(sampled));
    return max(v, minimum);
}

__device__ inline double sample_gamma_mt(
    double alpha,
    std::uint64_t seed,
    std::uint64_t scenario_id,
    std::uint64_t stream_base,
    std::uint64_t* counter
) {
    if (alpha <= 0.0) {
        return 0.0;
    }

    double shape = alpha;
    double scale_adjust = 1.0;
    if (shape < 1.0) {
        double u = fmax(random_u01_counter(seed, scenario_id, stream_base, counter), 1e-12);
        scale_adjust = pow(u, 1.0 / shape);
        shape += 1.0;
    }

    double d = shape - (1.0 / 3.0);
    double c = 1.0 / sqrt(9.0 * d);

    for (int attempt = 0; attempt < 256; ++attempt) {
        double x = random_normal_counter(seed, scenario_id, stream_base, counter);
        double v = 1.0 + c * x;
        if (v <= 0.0) {
            continue;
        }
        v = v * v * v;
        double u = random_u01_counter(seed, scenario_id, stream_base, counter);
        double x2 = x * x;
        if (u < 1.0 - 0.0331 * x2 * x2) {
            return d * v * scale_adjust;
        }
        if (log(u) < 0.5 * x2 + d * (1.0 - v + log(v))) {
            return d * v * scale_adjust;
        }
    }

    // Fallback keeps execution stable if acceptance is unexpectedly poor.
    return alpha * scale_adjust;
}

__device__ inline double sample_beta_scaled(
    double alpha,
    double beta,
    double low,
    double high,
    std::uint64_t seed,
    std::uint64_t scenario_id,
    std::uint64_t stream_base
) {
    if (alpha <= 0.0 || beta <= 0.0) {
        return clamp_probability(low);
    }

    std::uint64_t counter = 0ULL;
    double x = sample_gamma_mt(alpha, seed, scenario_id, stream_base, &counter);
    double y = sample_gamma_mt(beta, seed, scenario_id, stream_base, &counter);
    double sum = x + y;
    if (sum <= 0.0) {
        return clamp_probability(low);
    }

    double lo = fmin(low, high);
    double hi = fmax(low, high);
    double beta_u = x / sum;
    return clamp_probability(lo + beta_u * (hi - lo));
}

__device__ inline double sample_probability(
    int mode,
    double a,
    double b,
    double c,
    double d,
    std::uint64_t seed,
    std::uint64_t scenario_id,
    std::uint64_t stream_base
) {
    if (mode == 0) {
        return clamp_probability(a);
    }
    if (mode == 1 || mode == 2) {
        double u = random_u01(seed, scenario_id, stream_base);
        return clamp_probability(sample_continuous(mode, a, b, c, u));
    }
    if (mode == 3) {
        return sample_beta_scaled(a, b, c, d, seed, scenario_id, stream_base);
    }
    return clamp_probability(a);
}

__device__ inline double sell_stock_for_cash_need(
    double need,
    double* stocks,
    double* basis,
    double market_index,
    double ltcg_tax_rate,
    bool* unmet_need_out
) {
    if (need <= 1e-9) {
        return 0.0;
    }
    if (*stocks <= 1e-12) {
        if (unmet_need_out != nullptr) {
            *unmet_need_out = true;
        }
        return 0.0;
    }

    double basis_per_unit = *basis / *stocks;
    double gain_per_unit = fmax(0.0, market_index - basis_per_unit);
    double tax_per_unit = gain_per_unit * ltcg_tax_rate;
    double net_per_unit = fmax(1e-12, market_index - tax_per_unit);

    double units_to_sell = need / net_per_unit;
    if (units_to_sell > *stocks) {
        units_to_sell = *stocks;
    }

    double proceeds = units_to_sell * net_per_unit;
    double remaining_need = need - proceeds;
    if (remaining_need > 1e-9 && unmet_need_out != nullptr) {
        *unmet_need_out = true;
    }

    double prev_stocks = *stocks;
    double s_new = prev_stocks - units_to_sell;
    double b_new = *basis * (s_new / prev_stocks);
    *stocks = s_new;
    *basis = b_new;
    return proceeds;
}

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
    int retirement_enabled,
    int retirement_start_month_from_start,
    double retirement_safe_withdrawal_rate_annual,
    double retirement_expense_reduction_fraction,
    int retirement_dynamic_safe_withdrawal_rate,
    int reinvest_enabled,
    double reinvest_crash_drawdown_threshold,
    double reinvest_recovery_fraction_of_peak,
    double reinvest_fraction_of_initial_sale_proceeds,
    int reinvest_cash_buffer_months,
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
    double remaining_reinvest_budget = gross_sold - tax_paid;
    bool reinvest_done = false;
    double peak_index = 1.0;
    bool in_crash = false;
    double crash_peak_index = 1.0;
    double crash_trough_index = 1.0;

    for (int month_idx = 0; month_idx < horizon; ++month_idx) {
        int offset = scenario_idx * horizon + month_idx;
        double monthly_return = static_cast<double>(monthly_returns[offset]);
        market_index *= (1.0 + monthly_return);
        cash *= (1.0 + r_cash);
        peak_index = fmax(peak_index, market_index);

        if (reinvest_enabled != 0 && !reinvest_done) {
            double drawdown = fmax(0.0, 1.0 - (market_index / fmax(peak_index, 1e-12)));
            if (!in_crash && drawdown >= reinvest_crash_drawdown_threshold) {
                in_crash = true;
                crash_peak_index = peak_index;
                crash_trough_index = market_index;
            }
            if (in_crash) {
                crash_trough_index = fmin(crash_trough_index, market_index);
            }
            if (in_crash && market_index >= crash_peak_index * reinvest_recovery_fraction_of_peak) {
                bool retirement_active_for_buffer = retirement_enabled != 0
                    && retirement_start_month_from_start >= 0
                    && month_idx >= retirement_start_month_from_start;
                double expense_for_buffer = monthly_expenses;
                if (retirement_active_for_buffer) {
                    expense_for_buffer *= retirement_expense_reduction_fraction;
                }
                double cash_buffer_required =
                    static_cast<double>(reinvest_cash_buffer_months) * expense_for_buffer;
                double available_cash = fmax(0.0, cash - cash_buffer_required);
                double target_amount =
                    reinvest_fraction_of_initial_sale_proceeds * remaining_reinvest_budget;
                double reinvest_amount = fmin(
                    fmin(target_amount, remaining_reinvest_budget),
                    available_cash
                );
                if (reinvest_amount > 1e-9) {
                    stocks += reinvest_amount / market_index;
                    basis += reinvest_amount;
                    cash -= reinvest_amount;
                    remaining_reinvest_budget = fmax(0.0, remaining_reinvest_budget - reinvest_amount);
                }
                reinvest_done = true;
                in_crash = false;
            }
        }

        bool retirement_active = retirement_enabled != 0
            && retirement_start_month_from_start >= 0
            && month_idx >= retirement_start_month_from_start;

        bool is_unemployed = retirement_active ? true : (unemployment_matrix[offset] != 0);
        if (!is_unemployed) {
            stocks += monthly_savings / market_index;
            basis += monthly_savings;
        }

        double expense_this_month = monthly_expenses;
        if (retirement_active) {
            expense_this_month *= retirement_expense_reduction_fraction;
            double withdrawal_target = 0.0;
            if (retirement_dynamic_safe_withdrawal_rate != 0) {
                withdrawal_target = expense_this_month;
            } else {
                withdrawal_target = fmax(
                    0.0,
                    stocks * market_index * (retirement_safe_withdrawal_rate_annual / 12.0)
                );
            }
            double withdrawal_proceeds = sell_stock_for_cash_need(
                withdrawal_target,
                &stocks,
                &basis,
                market_index,
                ltcg_tax_rate,
                nullptr
            );
            cash += withdrawal_proceeds;
        }

        double need = is_unemployed ? expense_this_month : 0.0;
        double cash_before_expense = cash;
        double from_cash = fmin(need, cash);
        cash -= from_cash;
        need -= from_cash;
        if (reinvest_enabled != 0 && cash_before_expense > 1e-12 && remaining_reinvest_budget > 0.0) {
            double budget_share = remaining_reinvest_budget / cash_before_expense;
            budget_share = fmin(1.0, fmax(0.0, budget_share));
            remaining_reinvest_budget = fmax(
                0.0,
                remaining_reinvest_budget - (from_cash * budget_share)
            );
        }

        if (need > 1e-9) {
            bool unmet_need = false;
            sell_stock_for_cash_need(
                need,
                &stocks,
                &basis,
                market_index,
                ltcg_tax_rate,
                &unmet_need
            );
            if (unmet_need) {
                ruined = true;
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

__device__ inline void simulate_one_path_from_params(
    const std::uint8_t* is_crash,
    const int* crash_start,
    const int* down_end,
    const int* crash_end,
    const double* mu_down,
    const double* mu_up,
    const int* layoff_start,
    const int* layoff_end,
    int scenario_idx,
    const SyntheticGenerationParams& params,
    double fraction_sold,
    double initial_portfolio,
    double initial_basis,
    double ltcg_tax_rate,
    double monthly_expenses,
    double monthly_savings,
    int retirement_enabled,
    int retirement_start_month_from_start,
    double retirement_safe_withdrawal_rate_annual,
    double retirement_expense_reduction_fraction,
    int retirement_dynamic_safe_withdrawal_rate,
    int reinvest_enabled,
    double reinvest_crash_drawdown_threshold,
    double reinvest_recovery_fraction_of_peak,
    double reinvest_fraction_of_initial_sale_proceeds,
    int reinvest_cash_buffer_months,
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
    double remaining_reinvest_budget = gross_sold - tax_paid;
    bool reinvest_done = false;
    double peak_index = 1.0;
    bool in_crash = false;
    double crash_peak_index = 1.0;
    double crash_trough_index = 1.0;

    int s_crash_start = crash_start[scenario_idx];
    int s_down_end = down_end[scenario_idx];
    int s_crash_end = crash_end[scenario_idx];
    int s_layoff_start = layoff_start[scenario_idx];
    int s_layoff_end = layoff_end[scenario_idx];
    bool s_is_crash = is_crash[scenario_idx] != 0;

    std::uint64_t global_scenario = params.global_offset + static_cast<std::uint64_t>(scenario_idx);

    for (int month_idx = 0; month_idx < params.horizon; ++month_idx) {
        int month = month_idx + 1;
        double drift = params.monthly_market_drift;
        if (s_is_crash) {
            if (month >= s_crash_start && month < s_down_end) {
                drift = mu_down[scenario_idx];
            } else if (month >= s_down_end && month < s_crash_end) {
                drift = mu_up[scenario_idx];
            }
        }

        double z = random_normal(params.seed, global_scenario, month);
        double monthly_return = params.sigma * z + drift;
        monthly_return = fmin(1.5, fmax(-0.95, monthly_return));

        market_index *= (1.0 + monthly_return);
        cash *= (1.0 + r_cash);
        peak_index = fmax(peak_index, market_index);

        if (reinvest_enabled != 0 && !reinvest_done) {
            double drawdown = fmax(0.0, 1.0 - (market_index / fmax(peak_index, 1e-12)));
            if (!in_crash && drawdown >= reinvest_crash_drawdown_threshold) {
                in_crash = true;
                crash_peak_index = peak_index;
                crash_trough_index = market_index;
            }
            if (in_crash) {
                crash_trough_index = fmin(crash_trough_index, market_index);
            }
            if (in_crash && market_index >= crash_peak_index * reinvest_recovery_fraction_of_peak) {
                bool retirement_active_for_buffer = retirement_enabled != 0
                    && retirement_start_month_from_start >= 0
                    && month_idx >= retirement_start_month_from_start;
                double expense_for_buffer = monthly_expenses;
                if (retirement_active_for_buffer) {
                    expense_for_buffer *= retirement_expense_reduction_fraction;
                }
                double cash_buffer_required =
                    static_cast<double>(reinvest_cash_buffer_months) * expense_for_buffer;
                double available_cash = fmax(0.0, cash - cash_buffer_required);
                double target_amount =
                    reinvest_fraction_of_initial_sale_proceeds * remaining_reinvest_budget;
                double reinvest_amount = fmin(
                    fmin(target_amount, remaining_reinvest_budget),
                    available_cash
                );
                if (reinvest_amount > 1e-9) {
                    stocks += reinvest_amount / market_index;
                    basis += reinvest_amount;
                    cash -= reinvest_amount;
                    remaining_reinvest_budget = fmax(0.0, remaining_reinvest_budget - reinvest_amount);
                }
                reinvest_done = true;
                in_crash = false;
            }
        }

        bool retirement_active = retirement_enabled != 0
            && retirement_start_month_from_start >= 0
            && month_idx >= retirement_start_month_from_start;

        bool unemployed = retirement_active
            ? true
            : ((month >= s_layoff_start) && (month < s_layoff_end));
        if (!unemployed) {
            stocks += monthly_savings / market_index;
            basis += monthly_savings;
        }

        double expense_this_month = monthly_expenses;
        if (retirement_active) {
            expense_this_month *= retirement_expense_reduction_fraction;
            double withdrawal_target = 0.0;
            if (retirement_dynamic_safe_withdrawal_rate != 0) {
                withdrawal_target = expense_this_month;
            } else {
                withdrawal_target = fmax(
                    0.0,
                    stocks * market_index * (retirement_safe_withdrawal_rate_annual / 12.0)
                );
            }
            double withdrawal_proceeds = sell_stock_for_cash_need(
                withdrawal_target,
                &stocks,
                &basis,
                market_index,
                ltcg_tax_rate,
                nullptr
            );
            cash += withdrawal_proceeds;
        }

        double need = unemployed ? expense_this_month : 0.0;
        double cash_before_expense = cash;
        double from_cash = fmin(need, cash);
        cash -= from_cash;
        need -= from_cash;
        if (reinvest_enabled != 0 && cash_before_expense > 1e-12 && remaining_reinvest_budget > 0.0) {
            double budget_share = remaining_reinvest_budget / cash_before_expense;
            budget_share = fmin(1.0, fmax(0.0, budget_share));
            remaining_reinvest_budget = fmax(
                0.0,
                remaining_reinvest_budget - (from_cash * budget_share)
            );
        }

        if (need > 1e-9) {
            bool unmet_need = false;
            sell_stock_for_cash_need(
                need,
                &stocks,
                &basis,
                market_index,
                ltcg_tax_rate,
                &unmet_need
            );
            if (unmet_need) {
                ruined = true;
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

__global__ void generate_scenario_params_kernel(
    int n_sims,
    SyntheticGenerationParams params,
    std::uint8_t* out_is_crash,
    int* out_crash_start,
    int* out_down_end,
    int* out_crash_end,
    double* out_mu_down,
    double* out_mu_up,
    int* out_layoff_start,
    int* out_layoff_end,
    unsigned long long* out_summary_counts,
    double* out_summary_sums
) {
    int scenario_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (scenario_idx >= n_sims) {
        return;
    }

    std::uint64_t global_scenario = params.global_offset + static_cast<std::uint64_t>(scenario_idx);

    double prob_crash = sample_probability(
        params.prob_crash_mode,
        params.prob_crash_a,
        params.prob_crash_b,
        params.prob_crash_c,
        params.prob_crash_d,
        params.seed,
        global_scenario,
        1000000ULL
    );
    bool is_crash = random_u01(params.seed, global_scenario, 0ULL) < prob_crash;

    int crash_start_v = sample_positive_int(
        params.crash_start_mode,
        params.crash_start_a,
        params.crash_start_b,
        params.crash_start_c,
        random_u01(params.seed, global_scenario, 1ULL),
        1
    );
    int crash_duration_v = sample_positive_int(
        params.crash_duration_mode,
        params.crash_duration_a,
        params.crash_duration_b,
        params.crash_duration_c,
        random_u01(params.seed, global_scenario, 2ULL),
        1
    );
    double crash_severity_v = sample_continuous(
        params.crash_severity_mode,
        params.crash_severity_a,
        params.crash_severity_b,
        params.crash_severity_c,
        random_u01(params.seed, global_scenario, 3ULL)
    );
    crash_severity_v = fmin(0.95, fmax(1e-4, crash_severity_v));

    int down_months = max(1, crash_duration_v / 2);
    int up_months = max(1, crash_duration_v - down_months);
    int down_end_v = crash_start_v + down_months;
    int crash_end_v = crash_start_v + crash_duration_v;

    double mu_down_v = (pow(1.0 - crash_severity_v, 1.0 / static_cast<double>(down_months)) - 1.0);
    double mu_up_v = (pow(1.0 / (1.0 - crash_severity_v), 1.0 / static_cast<double>(up_months)) - 1.0);

    int layoff_start_v = params.horizon + 1;
    int layoff_end_v = params.horizon + 1;
    bool will_lose_job = false;

    if (is_crash) {
        double event_prob = clamp_probability(params.prob_layoff_during_crash);
        if (params.layoff_probabilities_are_annual != 0) {
            double years = static_cast<double>(crash_end_v - crash_start_v) / 12.0;
            event_prob = 1.0 - pow(1.0 - event_prob, years);
            event_prob = clamp_probability(event_prob);
        }

        if (random_u01(params.seed, global_scenario, 4ULL) < event_prob) {
            int latest_start = crash_end_v - params.crash_layoff_min_duration;
            if (latest_start >= crash_start_v) {
                int start_window = latest_start - crash_start_v + 1;
                int start_offset = static_cast<int>(floor(random_u01(params.seed, global_scenario, 5ULL) * start_window));
                int start_v = crash_start_v + start_offset;
                int max_duration = crash_end_v - start_v;
                int duration_span = max_duration - params.crash_layoff_min_duration + 1;
                if (duration_span > 0) {
                    int duration_offset = static_cast<int>(
                        floor(random_u01(params.seed, global_scenario, 6ULL) * duration_span)
                    );
                    int duration_v = params.crash_layoff_min_duration + duration_offset;
                    layoff_start_v = start_v;
                    layoff_end_v = start_v + duration_v;
                    will_lose_job = true;
                }
            }
        }
    } else {
        double event_prob = clamp_probability(params.prob_layoff_baseline);
        if (params.layoff_probabilities_are_annual != 0) {
            double years = static_cast<double>(params.horizon) / 12.0;
            event_prob = 1.0 - pow(1.0 - event_prob, years);
            event_prob = clamp_probability(event_prob);
        }

        if (random_u01(params.seed, global_scenario, 7ULL) < event_prob) {
            int latest_start = params.horizon - params.baseline_layoff_min_duration + 1;
            if (latest_start >= 1) {
                int start_v = 1 + static_cast<int>(
                    floor(random_u01(params.seed, global_scenario, 8ULL) * latest_start)
                );
                int max_duration = min(
                    params.baseline_layoff_max_duration,
                    params.horizon - start_v + 1
                );
                int duration_span = max_duration - params.baseline_layoff_min_duration + 1;
                if (duration_span > 0) {
                    int duration_offset = static_cast<int>(
                        floor(random_u01(params.seed, global_scenario, 9ULL) * duration_span)
                    );
                    int duration_v = params.baseline_layoff_min_duration + duration_offset;
                    layoff_start_v = start_v;
                    layoff_end_v = start_v + duration_v;
                    will_lose_job = true;
                }
            }
        }
    }

    if (!is_crash) {
        crash_start_v = params.horizon + 1;
        down_end_v = params.horizon + 1;
        crash_end_v = params.horizon + 1;
        mu_down_v = 0.0;
        mu_up_v = 0.0;
    }

    out_is_crash[scenario_idx] = is_crash ? 1 : 0;
    out_crash_start[scenario_idx] = crash_start_v;
    out_down_end[scenario_idx] = down_end_v;
    out_crash_end[scenario_idx] = crash_end_v;
    out_mu_down[scenario_idx] = mu_down_v;
    out_mu_up[scenario_idx] = mu_up_v;
    out_layoff_start[scenario_idx] = layoff_start_v;
    out_layoff_end[scenario_idx] = layoff_end_v;

    if (out_summary_counts != nullptr && out_summary_sums != nullptr) {
        if (is_crash) {
            atomicAdd(&out_summary_counts[0], 1ULL);
            atomicAdd(&out_summary_sums[0], static_cast<double>(crash_start_v));
            atomicAdd(&out_summary_sums[1], static_cast<double>(crash_duration_v));
            atomicAdd(&out_summary_sums[2], crash_severity_v);
            if (will_lose_job) {
                atomicAdd(&out_summary_counts[1], 1ULL);
            }
        } else {
            atomicAdd(&out_summary_counts[2], 1ULL);
            if (will_lose_job) {
                atomicAdd(&out_summary_counts[3], 1ULL);
            }
        }
    }
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
    int retirement_enabled,
    int retirement_start_month_from_start,
    double retirement_safe_withdrawal_rate_annual,
    double retirement_expense_reduction_fraction,
    int retirement_dynamic_safe_withdrawal_rate,
    int reinvest_enabled,
    double reinvest_crash_drawdown_threshold,
    double reinvest_recovery_fraction_of_peak,
    double reinvest_fraction_of_initial_sale_proceeds,
    int reinvest_cash_buffer_months,
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
    int retirement_enabled,
    int retirement_start_month_from_start,
    double retirement_safe_withdrawal_rate_annual,
    double retirement_expense_reduction_fraction,
    int retirement_dynamic_safe_withdrawal_rate,
    int reinvest_enabled,
    double reinvest_crash_drawdown_threshold,
    double reinvest_recovery_fraction_of_peak,
    double reinvest_fraction_of_initial_sale_proceeds,
    int reinvest_cash_buffer_months,
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

__global__ void simulate_fraction_tile_reduced_from_params_kernel(
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
    int fraction_start_idx,
    SyntheticGenerationParams params,
    double initial_portfolio,
    double initial_basis,
    double ltcg_tax_rate,
    double monthly_expenses,
    double monthly_savings,
    int retirement_enabled,
    int retirement_start_month_from_start,
    double retirement_safe_withdrawal_rate_annual,
    double retirement_expense_reduction_fraction,
    int retirement_dynamic_safe_withdrawal_rate,
    int reinvest_enabled,
    double reinvest_crash_drawdown_threshold,
    double reinvest_recovery_fraction_of_peak,
    double reinvest_fraction_of_initial_sale_proceeds,
    int reinvest_cash_buffer_months,
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
    int fraction_idx = fraction_start_idx + local_fraction_idx;
    int scenario_idx = blockIdx.x * blockDim.x + threadIdx.x;

    double final_net_worth = 0.0;
    double log_score = 0.0;
    unsigned int ruined = 0;

    if (scenario_idx < n_sims) {
        bool scenario_ruined = false;
        simulate_one_path_from_params(
            is_crash,
            crash_start,
            down_end,
            crash_end,
            mu_down,
            mu_up,
            layoff_start,
            layoff_end,
            scenario_idx,
            params,
            fractions[fraction_idx],
            initial_portfolio,
            initial_basis,
            ltcg_tax_rate,
            monthly_expenses,
            monthly_savings,
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

__global__ void simulate_fraction_tile_full_from_params_kernel(
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
    SyntheticGenerationParams params,
    double initial_portfolio,
    double initial_basis,
    double ltcg_tax_rate,
    double monthly_expenses,
    double monthly_savings,
    int retirement_enabled,
    int retirement_start_month_from_start,
    double retirement_safe_withdrawal_rate_annual,
    double retirement_expense_reduction_fraction,
    int retirement_dynamic_safe_withdrawal_rate,
    int reinvest_enabled,
    double reinvest_crash_drawdown_threshold,
    double reinvest_recovery_fraction_of_peak,
    double reinvest_fraction_of_initial_sale_proceeds,
    int reinvest_cash_buffer_months,
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
    simulate_one_path_from_params(
        is_crash,
        crash_start,
        down_end,
        crash_end,
        mu_down,
        mu_up,
        layoff_start,
        layoff_end,
        scenario_idx,
        params,
        fractions[fraction_idx],
        initial_portfolio,
        initial_basis,
        ltcg_tax_rate,
        monthly_expenses,
        monthly_savings,
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
        cash_yield_annual,
        &final_net_worth,
        &ruined
    );

    int out_idx = fraction_idx * n_sims + scenario_idx;
    out_final_net_worth[out_idx] = final_net_worth;
    out_ruined[out_idx] = ruined ? 1 : 0;
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
    int retirement_enabled,
    int retirement_start_month_from_start,
    double retirement_safe_withdrawal_rate_annual,
    double retirement_expense_reduction_fraction,
    int retirement_dynamic_safe_withdrawal_rate,
    int reinvest_enabled,
    double reinvest_crash_drawdown_threshold,
    double reinvest_recovery_fraction_of_peak,
    double reinvest_fraction_of_initial_sale_proceeds,
    int reinvest_cash_buffer_months,
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
    int retirement_enabled,
    int retirement_start_month_from_start,
    double retirement_safe_withdrawal_rate_annual,
    double retirement_expense_reduction_fraction,
    int retirement_dynamic_safe_withdrawal_rate,
    int reinvest_enabled,
    double reinvest_crash_drawdown_threshold,
    double reinvest_recovery_fraction_of_peak,
    double reinvest_fraction_of_initial_sale_proceeds,
    int reinvest_cash_buffer_months,
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
) {
    int threads = 256;
    int blocks = (n_sims + threads - 1) / threads;
    generate_scenario_params_kernel<<<blocks, threads, 0, stream>>>(
        n_sims,
        params,
        out_is_crash,
        out_crash_start,
        out_down_end,
        out_crash_end,
        out_mu_down,
        out_mu_up,
        out_layoff_start,
        out_layoff_end,
        out_summary_counts,
        out_summary_sums
    );
}

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
    int retirement_enabled,
    int retirement_start_month_from_start,
    double retirement_safe_withdrawal_rate_annual,
    double retirement_expense_reduction_fraction,
    int retirement_dynamic_safe_withdrawal_rate,
    int reinvest_enabled,
    double reinvest_crash_drawdown_threshold,
    double reinvest_recovery_fraction_of_peak,
    double reinvest_fraction_of_initial_sale_proceeds,
    int reinvest_cash_buffer_months,
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
    if (fraction_count <= 0 || n_sims <= 0 || params.horizon <= 0) {
        return;
    }

    int threads = 256;
    int blocks_x = (n_sims + threads - 1) / threads;
    dim3 grid(static_cast<unsigned int>(blocks_x), static_cast<unsigned int>(fraction_count), 1u);
    std::size_t shared_bytes = static_cast<std::size_t>(threads) * (2 * sizeof(double) + sizeof(unsigned int));

    simulate_fraction_tile_reduced_from_params_kernel<<<grid, threads, shared_bytes, stream>>>(
        is_crash,
        crash_start,
        down_end,
        crash_end,
        mu_down,
        mu_up,
        layoff_start,
        layoff_end,
        fractions,
        n_sims,
        fraction_start,
        params,
        initial_portfolio,
        initial_basis,
        ltcg_tax_rate,
        monthly_expenses,
        monthly_savings,
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
    int retirement_enabled,
    int retirement_start_month_from_start,
    double retirement_safe_withdrawal_rate_annual,
    double retirement_expense_reduction_fraction,
    int retirement_dynamic_safe_withdrawal_rate,
    int reinvest_enabled,
    double reinvest_crash_drawdown_threshold,
    double reinvest_recovery_fraction_of_peak,
    double reinvest_fraction_of_initial_sale_proceeds,
    int reinvest_cash_buffer_months,
    double cash_yield_annual,
    double* out_final_net_worth,
    std::uint8_t* out_ruined,
    cudaStream_t stream
) {
    int total = n_sims * n_fractions;
    if (total <= 0 || params.horizon <= 0) {
        return;
    }

    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    simulate_fraction_tile_full_from_params_kernel<<<blocks, threads, 0, stream>>>(
        is_crash,
        crash_start,
        down_end,
        crash_end,
        mu_down,
        mu_up,
        layoff_start,
        layoff_end,
        fractions,
        n_sims,
        n_fractions,
        params,
        initial_portfolio,
        initial_basis,
        ltcg_tax_rate,
        monthly_expenses,
        monthly_savings,
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
        cash_yield_annual,
        out_final_net_worth,
        out_ruined
    );
}
