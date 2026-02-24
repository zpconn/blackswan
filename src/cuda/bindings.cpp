#include <algorithm>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "sim_kernel.cuh"

namespace py = pybind11;

namespace {

void check_cuda(cudaError_t status, const std::string& message) {
    if (status != cudaSuccess) {
        throw std::runtime_error(message + ": " + cudaGetErrorString(status));
    }
}

template <typename T>
T dict_get_required(const py::dict& d, const char* key) {
    py::str k(key);
    if (!d.contains(k)) {
        throw std::runtime_error(std::string("Missing required synthetic param: ") + key);
    }
    return py::cast<T>(d[k]);
}

SyntheticGenerationParams parse_synthetic_params(const py::dict& params_dict) {
    SyntheticGenerationParams params{};
    params.seed = dict_get_required<std::uint64_t>(params_dict, "seed");
    params.global_offset = dict_get_required<std::uint64_t>(params_dict, "global_offset");
    params.horizon = dict_get_required<int>(params_dict, "horizon");
    params.sigma = dict_get_required<double>(params_dict, "sigma");
    params.monthly_market_drift = dict_get_required<double>(params_dict, "monthly_market_drift");
    py::str prob_mode_key("prob_crash_mode");
    if (params_dict.contains(prob_mode_key)) {
        params.prob_crash_mode = dict_get_required<int>(params_dict, "prob_crash_mode");
        params.prob_crash_a = dict_get_required<double>(params_dict, "prob_crash_a");
        params.prob_crash_b = dict_get_required<double>(params_dict, "prob_crash_b");
        params.prob_crash_c = dict_get_required<double>(params_dict, "prob_crash_c");
        params.prob_crash_d = dict_get_required<double>(params_dict, "prob_crash_d");
    } else {
        // Backward compatibility for older Python callers.
        double scalar_prob_crash = dict_get_required<double>(params_dict, "prob_crash");
        params.prob_crash_mode = 0;
        params.prob_crash_a = scalar_prob_crash;
        params.prob_crash_b = scalar_prob_crash;
        params.prob_crash_c = scalar_prob_crash;
        params.prob_crash_d = scalar_prob_crash;
    }

    params.crash_start_mode = dict_get_required<int>(params_dict, "crash_start_mode");
    params.crash_start_a = dict_get_required<double>(params_dict, "crash_start_a");
    params.crash_start_b = dict_get_required<double>(params_dict, "crash_start_b");
    params.crash_start_c = dict_get_required<double>(params_dict, "crash_start_c");

    params.crash_duration_mode = dict_get_required<int>(params_dict, "crash_duration_mode");
    params.crash_duration_a = dict_get_required<double>(params_dict, "crash_duration_a");
    params.crash_duration_b = dict_get_required<double>(params_dict, "crash_duration_b");
    params.crash_duration_c = dict_get_required<double>(params_dict, "crash_duration_c");

    params.crash_severity_mode = dict_get_required<int>(params_dict, "crash_severity_mode");
    params.crash_severity_a = dict_get_required<double>(params_dict, "crash_severity_a");
    params.crash_severity_b = dict_get_required<double>(params_dict, "crash_severity_b");
    params.crash_severity_c = dict_get_required<double>(params_dict, "crash_severity_c");

    params.prob_layoff_during_crash = dict_get_required<double>(params_dict, "prob_layoff_during_crash");
    params.prob_layoff_baseline = dict_get_required<double>(params_dict, "prob_layoff_baseline");
    params.layoff_probabilities_are_annual = dict_get_required<int>(params_dict, "layoff_probabilities_are_annual");
    params.crash_layoff_min_duration = dict_get_required<int>(params_dict, "crash_layoff_min_duration");
    params.baseline_layoff_min_duration = dict_get_required<int>(params_dict, "baseline_layoff_min_duration");
    params.baseline_layoff_max_duration = dict_get_required<int>(params_dict, "baseline_layoff_max_duration");

    return params;
}

struct ScenarioDeviceBuffers {
    std::uint8_t* is_crash = nullptr;
    int* crash_start = nullptr;
    int* down_end = nullptr;
    int* crash_end = nullptr;
    double* mu_down = nullptr;
    double* mu_up = nullptr;
    int* layoff_start = nullptr;
    int* layoff_end = nullptr;
    unsigned long long* summary_counts = nullptr;
    double* summary_sums = nullptr;
};

void free_scenario_buffers(ScenarioDeviceBuffers& b) {
    cudaFree(b.is_crash);
    cudaFree(b.crash_start);
    cudaFree(b.down_end);
    cudaFree(b.crash_end);
    cudaFree(b.mu_down);
    cudaFree(b.mu_up);
    cudaFree(b.layoff_start);
    cudaFree(b.layoff_end);
    cudaFree(b.summary_counts);
    cudaFree(b.summary_sums);
    b = ScenarioDeviceBuffers{};
}

void allocate_scenario_buffers(int n_sims, bool with_summary, ScenarioDeviceBuffers& out) {
    check_cuda(cudaMalloc(&out.is_crash, static_cast<std::size_t>(n_sims) * sizeof(std::uint8_t)), "cudaMalloc(is_crash) failed");
    check_cuda(cudaMalloc(&out.crash_start, static_cast<std::size_t>(n_sims) * sizeof(int)), "cudaMalloc(crash_start) failed");
    check_cuda(cudaMalloc(&out.down_end, static_cast<std::size_t>(n_sims) * sizeof(int)), "cudaMalloc(down_end) failed");
    check_cuda(cudaMalloc(&out.crash_end, static_cast<std::size_t>(n_sims) * sizeof(int)), "cudaMalloc(crash_end) failed");
    check_cuda(cudaMalloc(&out.mu_down, static_cast<std::size_t>(n_sims) * sizeof(double)), "cudaMalloc(mu_down) failed");
    check_cuda(cudaMalloc(&out.mu_up, static_cast<std::size_t>(n_sims) * sizeof(double)), "cudaMalloc(mu_up) failed");
    check_cuda(cudaMalloc(&out.layoff_start, static_cast<std::size_t>(n_sims) * sizeof(int)), "cudaMalloc(layoff_start) failed");
    check_cuda(cudaMalloc(&out.layoff_end, static_cast<std::size_t>(n_sims) * sizeof(int)), "cudaMalloc(layoff_end) failed");

    if (with_summary) {
        check_cuda(cudaMalloc(&out.summary_counts, static_cast<std::size_t>(4) * sizeof(unsigned long long)), "cudaMalloc(summary_counts) failed");
        check_cuda(cudaMalloc(&out.summary_sums, static_cast<std::size_t>(3) * sizeof(double)), "cudaMalloc(summary_sums) failed");
        check_cuda(cudaMemset(out.summary_counts, 0, static_cast<std::size_t>(4) * sizeof(unsigned long long)), "cudaMemset(summary_counts) failed");
        check_cuda(cudaMemset(out.summary_sums, 0, static_cast<std::size_t>(3) * sizeof(double)), "cudaMemset(summary_sums) failed");
    }
}

py::dict copy_summary_to_host(int n_sims, const ScenarioDeviceBuffers& buffers) {
    std::vector<unsigned long long> counts(4, 0ULL);
    std::vector<double> sums(3, 0.0);
    check_cuda(
        cudaMemcpy(counts.data(), buffers.summary_counts, static_cast<std::size_t>(4) * sizeof(unsigned long long), cudaMemcpyDeviceToHost),
        "cudaMemcpy D2H summary_counts failed"
    );
    check_cuda(
        cudaMemcpy(sums.data(), buffers.summary_sums, static_cast<std::size_t>(3) * sizeof(double), cudaMemcpyDeviceToHost),
        "cudaMemcpy D2H summary_sums failed"
    );

    py::dict summary;
    summary["n"] = n_sims;
    summary["crash_count"] = static_cast<std::int64_t>(counts[0]);
    summary["crash_start_sum"] = sums[0];
    summary["crash_duration_sum"] = sums[1];
    summary["crash_severity_sum"] = sums[2];
    summary["layoff_crash_count"] = static_cast<std::int64_t>(counts[1]);
    summary["non_crash_count"] = static_cast<std::int64_t>(counts[2]);
    summary["layoff_non_crash_count"] = static_cast<std::int64_t>(counts[3]);
    return summary;
}

std::pair<py::array_t<double>, py::array_t<std::uint8_t>> simulate_fraction_tile(
    py::array_t<float, py::array::c_style | py::array::forcecast> monthly_returns,
    py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> unemployment_matrix,
    py::array_t<double, py::array::c_style | py::array::forcecast> fractions,
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
    double cash_yield_annual
) {
    auto returns_info = monthly_returns.request();
    auto unemployment_info = unemployment_matrix.request();
    auto fractions_info = fractions.request();

    if (returns_info.ndim != 2) {
        throw std::runtime_error("monthly_returns must be a 2D array.");
    }
    if (unemployment_info.ndim != 2) {
        throw std::runtime_error("unemployment_matrix must be a 2D array.");
    }
    if (fractions_info.ndim != 1) {
        throw std::runtime_error("fractions must be a 1D array.");
    }

    int n_sims = static_cast<int>(returns_info.shape[0]);
    int horizon = static_cast<int>(returns_info.shape[1]);
    if (unemployment_info.shape[0] != n_sims || unemployment_info.shape[1] != horizon) {
        throw std::runtime_error("unemployment_matrix shape must match monthly_returns.");
    }
    int n_fractions = static_cast<int>(fractions_info.shape[0]);
    if (n_fractions <= 0 || n_sims <= 0 || horizon <= 0) {
        throw std::runtime_error("Invalid shape sizes passed to CUDA simulation.");
    }

    py::array_t<double> final_net_worth({n_fractions, n_sims});
    py::array_t<std::uint8_t> ruined({n_fractions, n_sims});
    auto final_info = final_net_worth.request();
    auto ruined_info = ruined.request();

    const float* h_monthly_returns = static_cast<const float*>(returns_info.ptr);
    const std::uint8_t* h_unemployment = static_cast<const std::uint8_t*>(unemployment_info.ptr);
    const double* h_fractions = static_cast<const double*>(fractions_info.ptr);
    double* h_final = static_cast<double*>(final_info.ptr);
    std::uint8_t* h_ruined = static_cast<std::uint8_t*>(ruined_info.ptr);

    float* d_monthly_returns = nullptr;
    std::uint8_t* d_unemployment = nullptr;
    double* d_fractions = nullptr;
    double* d_final = nullptr;
    std::uint8_t* d_ruined = nullptr;

    std::size_t monthly_bytes = static_cast<std::size_t>(n_sims) * horizon * sizeof(float);
    std::size_t unemployment_bytes = static_cast<std::size_t>(n_sims) * horizon * sizeof(std::uint8_t);
    std::size_t fractions_bytes = static_cast<std::size_t>(n_fractions) * sizeof(double);
    std::size_t out_double_bytes = static_cast<std::size_t>(n_fractions) * n_sims * sizeof(double);
    std::size_t out_ruined_bytes = static_cast<std::size_t>(n_fractions) * n_sims * sizeof(std::uint8_t);

    try {
        check_cuda(cudaMalloc(&d_monthly_returns, monthly_bytes), "cudaMalloc(monthly_returns) failed");
        check_cuda(cudaMalloc(&d_unemployment, unemployment_bytes), "cudaMalloc(unemployment_matrix) failed");
        check_cuda(cudaMalloc(&d_fractions, fractions_bytes), "cudaMalloc(fractions) failed");
        check_cuda(cudaMalloc(&d_final, out_double_bytes), "cudaMalloc(final) failed");
        check_cuda(cudaMalloc(&d_ruined, out_ruined_bytes), "cudaMalloc(ruined) failed");

        check_cuda(
            cudaMemcpy(d_monthly_returns, h_monthly_returns, monthly_bytes, cudaMemcpyHostToDevice),
            "cudaMemcpy H2D monthly_returns failed"
        );
        check_cuda(
            cudaMemcpy(d_unemployment, h_unemployment, unemployment_bytes, cudaMemcpyHostToDevice),
            "cudaMemcpy H2D unemployment_matrix failed"
        );
        check_cuda(
            cudaMemcpy(d_fractions, h_fractions, fractions_bytes, cudaMemcpyHostToDevice),
            "cudaMemcpy H2D fractions failed"
        );

        launch_simulate_fraction_tile(
            d_monthly_returns,
            d_unemployment,
            d_fractions,
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
            d_final,
            d_ruined
        );

        check_cuda(cudaGetLastError(), "Kernel launch failed");
        check_cuda(cudaDeviceSynchronize(), "Kernel execution failed");

        check_cuda(
            cudaMemcpy(h_final, d_final, out_double_bytes, cudaMemcpyDeviceToHost),
            "cudaMemcpy D2H final failed"
        );
        check_cuda(
            cudaMemcpy(h_ruined, d_ruined, out_ruined_bytes, cudaMemcpyDeviceToHost),
            "cudaMemcpy D2H ruined failed"
        );
    } catch (...) {
        cudaFree(d_monthly_returns);
        cudaFree(d_unemployment);
        cudaFree(d_fractions);
        cudaFree(d_final);
        cudaFree(d_ruined);
        throw;
    }

    cudaFree(d_monthly_returns);
    cudaFree(d_unemployment);
    cudaFree(d_fractions);
    cudaFree(d_final);
    cudaFree(d_ruined);

    return {final_net_worth, ruined};
}

py::tuple simulate_fraction_tile_aggregates(
    py::array_t<float, py::array::c_style | py::array::forcecast> monthly_returns,
    py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> unemployment_matrix,
    py::array_t<double, py::array::c_style | py::array::forcecast> fractions,
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
    py::object sample_positions_obj,
    int streams
) {
    auto returns_info = monthly_returns.request();
    auto unemployment_info = unemployment_matrix.request();
    auto fractions_info = fractions.request();

    if (returns_info.ndim != 2) {
        throw std::runtime_error("monthly_returns must be a 2D array.");
    }
    if (unemployment_info.ndim != 2) {
        throw std::runtime_error("unemployment_matrix must be a 2D array.");
    }
    if (fractions_info.ndim != 1) {
        throw std::runtime_error("fractions must be a 1D array.");
    }

    int n_sims = static_cast<int>(returns_info.shape[0]);
    int horizon = static_cast<int>(returns_info.shape[1]);
    int n_fractions = static_cast<int>(fractions_info.shape[0]);
    if (unemployment_info.shape[0] != n_sims || unemployment_info.shape[1] != horizon) {
        throw std::runtime_error("unemployment_matrix shape must match monthly_returns.");
    }
    if (n_fractions <= 0 || n_sims <= 0 || horizon <= 0) {
        throw std::runtime_error("Invalid shape sizes passed to CUDA simulation.");
    }

    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> sample_positions;
    int sample_count = 0;
    if (!sample_positions_obj.is_none()) {
        sample_positions = sample_positions_obj.cast<py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>>();
        auto sample_info = sample_positions.request();
        if (sample_info.ndim != 1) {
            throw std::runtime_error("sample_positions must be a 1D int64 array or None.");
        }
        sample_count = static_cast<int>(sample_info.shape[0]);
    }

    py::array_t<double> sum_final({n_fractions});
    py::array_t<double> sum_log({n_fractions});
    py::array_t<std::int64_t> ruin_count({n_fractions});
    py::array_t<double> sampled_final({n_fractions, sample_count});

    auto sum_final_info = sum_final.request();
    auto sum_log_info = sum_log.request();
    auto ruin_count_info = ruin_count.request();
    auto sampled_final_info = sampled_final.request();

    const float* h_monthly_returns = static_cast<const float*>(returns_info.ptr);
    const std::uint8_t* h_unemployment = static_cast<const std::uint8_t*>(unemployment_info.ptr);
    const double* h_fractions = static_cast<const double*>(fractions_info.ptr);
    double* h_sum_final = static_cast<double*>(sum_final_info.ptr);
    double* h_sum_log = static_cast<double*>(sum_log_info.ptr);
    std::int64_t* h_ruin_count = static_cast<std::int64_t*>(ruin_count_info.ptr);
    double* h_sampled_final = static_cast<double*>(sampled_final_info.ptr);

    std::vector<int> h_sample_map;
    if (sample_count > 0) {
        auto sample_info = sample_positions.request();
        const std::int64_t* h_sample_positions = static_cast<const std::int64_t*>(sample_info.ptr);
        h_sample_map.assign(n_sims, -1);
        for (int i = 0; i < sample_count; ++i) {
            std::int64_t pos = h_sample_positions[i];
            if (pos < 0 || pos >= n_sims) {
                throw std::runtime_error("sample_positions contains index out of range for current chunk.");
            }
            h_sample_map[static_cast<int>(pos)] = i;
        }
    }

    float* d_monthly_returns = nullptr;
    std::uint8_t* d_unemployment = nullptr;
    double* d_fractions = nullptr;
    double* d_sum_final = nullptr;
    double* d_sum_log = nullptr;
    unsigned long long* d_ruin_count = nullptr;
    int* d_sample_map = nullptr;
    double* d_sampled_final = nullptr;
    std::vector<cudaStream_t> cuda_streams;

    std::size_t monthly_bytes = static_cast<std::size_t>(n_sims) * horizon * sizeof(float);
    std::size_t unemployment_bytes = static_cast<std::size_t>(n_sims) * horizon * sizeof(std::uint8_t);
    std::size_t fractions_bytes = static_cast<std::size_t>(n_fractions) * sizeof(double);
    std::size_t aggregate_bytes = static_cast<std::size_t>(n_fractions) * sizeof(double);
    std::size_t ruin_bytes = static_cast<std::size_t>(n_fractions) * sizeof(unsigned long long);
    std::size_t sample_map_bytes = static_cast<std::size_t>(n_sims) * sizeof(int);
    std::size_t sampled_bytes = static_cast<std::size_t>(n_fractions) * sample_count * sizeof(double);

    auto cleanup = [&]() {
        for (cudaStream_t stream : cuda_streams) {
            if (stream != nullptr) {
                cudaStreamDestroy(stream);
            }
        }
        cudaFree(d_monthly_returns);
        cudaFree(d_unemployment);
        cudaFree(d_fractions);
        cudaFree(d_sum_final);
        cudaFree(d_sum_log);
        cudaFree(d_ruin_count);
        cudaFree(d_sample_map);
        cudaFree(d_sampled_final);
    };

    double kernel_time_ms = 0.0;
    double transfer_time_ms = 0.0;

    try {
        check_cuda(cudaMalloc(&d_monthly_returns, monthly_bytes), "cudaMalloc(monthly_returns) failed");
        check_cuda(cudaMalloc(&d_unemployment, unemployment_bytes), "cudaMalloc(unemployment_matrix) failed");
        check_cuda(cudaMalloc(&d_fractions, fractions_bytes), "cudaMalloc(fractions) failed");
        check_cuda(cudaMalloc(&d_sum_final, aggregate_bytes), "cudaMalloc(sum_final) failed");
        check_cuda(cudaMalloc(&d_sum_log, aggregate_bytes), "cudaMalloc(sum_log) failed");
        check_cuda(cudaMalloc(&d_ruin_count, ruin_bytes), "cudaMalloc(ruin_count) failed");

        if (sample_count > 0) {
            check_cuda(cudaMalloc(&d_sample_map, sample_map_bytes), "cudaMalloc(sample_map) failed");
            check_cuda(cudaMalloc(&d_sampled_final, sampled_bytes), "cudaMalloc(sampled_final) failed");
        }

        auto t_transfer_start = std::chrono::steady_clock::now();
        check_cuda(
            cudaMemcpy(d_monthly_returns, h_monthly_returns, monthly_bytes, cudaMemcpyHostToDevice),
            "cudaMemcpy H2D monthly_returns failed"
        );
        check_cuda(
            cudaMemcpy(d_unemployment, h_unemployment, unemployment_bytes, cudaMemcpyHostToDevice),
            "cudaMemcpy H2D unemployment_matrix failed"
        );
        check_cuda(
            cudaMemcpy(d_fractions, h_fractions, fractions_bytes, cudaMemcpyHostToDevice),
            "cudaMemcpy H2D fractions failed"
        );
        check_cuda(cudaMemset(d_sum_final, 0, aggregate_bytes), "cudaMemset(sum_final) failed");
        check_cuda(cudaMemset(d_sum_log, 0, aggregate_bytes), "cudaMemset(sum_log) failed");
        check_cuda(cudaMemset(d_ruin_count, 0, ruin_bytes), "cudaMemset(ruin_count) failed");

        if (sample_count > 0) {
            check_cuda(
                cudaMemcpy(d_sample_map, h_sample_map.data(), sample_map_bytes, cudaMemcpyHostToDevice),
                "cudaMemcpy H2D sample_map failed"
            );
            check_cuda(cudaMemset(d_sampled_final, 0, sampled_bytes), "cudaMemset(sampled_final) failed");
        }
        auto t_transfer_end = std::chrono::steady_clock::now();
        transfer_time_ms += std::chrono::duration<double, std::milli>(t_transfer_end - t_transfer_start).count();

        int stream_count = std::max(1, std::min(streams, n_fractions));
        cuda_streams.resize(stream_count, nullptr);
        for (int i = 0; i < stream_count; ++i) {
            check_cuda(cudaStreamCreateWithFlags(&cuda_streams[i], cudaStreamNonBlocking), "cudaStreamCreate failed");
        }

        auto t_kernel_start = std::chrono::steady_clock::now();
        int base = n_fractions / stream_count;
        int remainder = n_fractions % stream_count;
        int fraction_start = 0;
        for (int i = 0; i < stream_count; ++i) {
            int count = base + (i < remainder ? 1 : 0);
            if (count <= 0) {
                continue;
            }
            launch_simulate_fraction_tile_aggregates(
                d_monthly_returns,
                d_unemployment,
                d_fractions,
                n_sims,
                horizon,
                fraction_start,
                count,
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
                d_sample_map,
                sample_count,
                d_sum_final,
                d_sum_log,
                d_ruin_count,
                d_sampled_final,
                cuda_streams[i]
            );
            check_cuda(cudaGetLastError(), "Kernel launch failed");
            fraction_start += count;
        }

        for (cudaStream_t stream : cuda_streams) {
            check_cuda(cudaStreamSynchronize(stream), "Kernel execution failed");
        }
        auto t_kernel_end = std::chrono::steady_clock::now();
        kernel_time_ms = std::chrono::duration<double, std::milli>(t_kernel_end - t_kernel_start).count();

        std::vector<unsigned long long> h_ruin_temp(static_cast<std::size_t>(n_fractions), 0ull);
        auto t_copy_back_start = std::chrono::steady_clock::now();
        check_cuda(
            cudaMemcpy(h_sum_final, d_sum_final, aggregate_bytes, cudaMemcpyDeviceToHost),
            "cudaMemcpy D2H sum_final failed"
        );
        check_cuda(
            cudaMemcpy(h_sum_log, d_sum_log, aggregate_bytes, cudaMemcpyDeviceToHost),
            "cudaMemcpy D2H sum_log failed"
        );
        check_cuda(
            cudaMemcpy(h_ruin_temp.data(), d_ruin_count, ruin_bytes, cudaMemcpyDeviceToHost),
            "cudaMemcpy D2H ruin_count failed"
        );
        if (sample_count > 0) {
            check_cuda(
                cudaMemcpy(h_sampled_final, d_sampled_final, sampled_bytes, cudaMemcpyDeviceToHost),
                "cudaMemcpy D2H sampled_final failed"
            );
        }
        auto t_copy_back_end = std::chrono::steady_clock::now();
        transfer_time_ms += std::chrono::duration<double, std::milli>(t_copy_back_end - t_copy_back_start).count();

        for (int i = 0; i < n_fractions; ++i) {
            h_ruin_count[i] = static_cast<std::int64_t>(h_ruin_temp[static_cast<std::size_t>(i)]);
        }
    } catch (...) {
        cleanup();
        throw;
    }

    cleanup();

    return py::make_tuple(sum_final, sum_log, ruin_count, sampled_final, kernel_time_ms, transfer_time_ms);
}

py::tuple simulate_fraction_tile_aggregates_synthetic(
    int n_sims,
    py::array_t<double, py::array::c_style | py::array::forcecast> fractions,
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
    py::dict synthetic_params,
    py::object sample_positions_obj,
    int streams
) {
    auto fractions_info = fractions.request();
    if (fractions_info.ndim != 1) {
        throw std::runtime_error("fractions must be a 1D array.");
    }
    int n_fractions = static_cast<int>(fractions_info.shape[0]);
    if (n_sims <= 0 || n_fractions <= 0) {
        throw std::runtime_error("Invalid n_sims or fraction count for synthetic aggregate path.");
    }

    SyntheticGenerationParams params = parse_synthetic_params(synthetic_params);
    if (params.horizon <= 0) {
        throw std::runtime_error("synthetic_params.horizon must be > 0.");
    }

    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> sample_positions;
    int sample_count = 0;
    if (!sample_positions_obj.is_none()) {
        sample_positions = sample_positions_obj.cast<py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>>();
        auto sample_info = sample_positions.request();
        if (sample_info.ndim != 1) {
            throw std::runtime_error("sample_positions must be a 1D int64 array or None.");
        }
        sample_count = static_cast<int>(sample_info.shape[0]);
    }

    py::array_t<double> sum_final({n_fractions});
    py::array_t<double> sum_log({n_fractions});
    py::array_t<std::int64_t> ruin_count({n_fractions});
    py::array_t<double> sampled_final({n_fractions, sample_count});

    auto sum_final_info = sum_final.request();
    auto sum_log_info = sum_log.request();
    auto ruin_count_info = ruin_count.request();
    auto sampled_final_info = sampled_final.request();

    const double* h_fractions = static_cast<const double*>(fractions_info.ptr);
    double* h_sum_final = static_cast<double*>(sum_final_info.ptr);
    double* h_sum_log = static_cast<double*>(sum_log_info.ptr);
    std::int64_t* h_ruin_count = static_cast<std::int64_t*>(ruin_count_info.ptr);
    double* h_sampled_final = static_cast<double*>(sampled_final_info.ptr);

    std::vector<int> h_sample_map;
    if (sample_count > 0) {
        auto sample_info = sample_positions.request();
        const std::int64_t* h_sample_positions = static_cast<const std::int64_t*>(sample_info.ptr);
        h_sample_map.assign(n_sims, -1);
        for (int i = 0; i < sample_count; ++i) {
            std::int64_t pos = h_sample_positions[i];
            if (pos < 0 || pos >= n_sims) {
                throw std::runtime_error("sample_positions contains index out of range for current chunk.");
            }
            h_sample_map[static_cast<int>(pos)] = i;
        }
    }

    ScenarioDeviceBuffers scenario{};
    double* d_fractions = nullptr;
    double* d_sum_final = nullptr;
    double* d_sum_log = nullptr;
    unsigned long long* d_ruin_count = nullptr;
    int* d_sample_map = nullptr;
    double* d_sampled_final = nullptr;
    std::vector<cudaStream_t> cuda_streams;
    cudaEvent_t scenario_ready_event = nullptr;

    std::size_t fractions_bytes = static_cast<std::size_t>(n_fractions) * sizeof(double);
    std::size_t aggregate_bytes = static_cast<std::size_t>(n_fractions) * sizeof(double);
    std::size_t ruin_bytes = static_cast<std::size_t>(n_fractions) * sizeof(unsigned long long);
    std::size_t sample_map_bytes = static_cast<std::size_t>(n_sims) * sizeof(int);
    std::size_t sampled_bytes = static_cast<std::size_t>(n_fractions) * sample_count * sizeof(double);

    auto cleanup = [&]() {
        for (cudaStream_t stream : cuda_streams) {
            if (stream != nullptr) {
                cudaStreamDestroy(stream);
            }
        }
        if (scenario_ready_event != nullptr) {
            cudaEventDestroy(scenario_ready_event);
        }
        cudaFree(d_fractions);
        cudaFree(d_sum_final);
        cudaFree(d_sum_log);
        cudaFree(d_ruin_count);
        cudaFree(d_sample_map);
        cudaFree(d_sampled_final);
        free_scenario_buffers(scenario);
    };

    double kernel_time_ms = 0.0;
    double transfer_time_ms = 0.0;
    py::dict summary;

    try {
        allocate_scenario_buffers(n_sims, true, scenario);
        check_cuda(cudaMalloc(&d_fractions, fractions_bytes), "cudaMalloc(fractions) failed");
        check_cuda(cudaMalloc(&d_sum_final, aggregate_bytes), "cudaMalloc(sum_final) failed");
        check_cuda(cudaMalloc(&d_sum_log, aggregate_bytes), "cudaMalloc(sum_log) failed");
        check_cuda(cudaMalloc(&d_ruin_count, ruin_bytes), "cudaMalloc(ruin_count) failed");

        if (sample_count > 0) {
            check_cuda(cudaMalloc(&d_sample_map, sample_map_bytes), "cudaMalloc(sample_map) failed");
            check_cuda(cudaMalloc(&d_sampled_final, sampled_bytes), "cudaMalloc(sampled_final) failed");
        }

        auto t_transfer_start = std::chrono::steady_clock::now();
        check_cuda(cudaMemcpy(d_fractions, h_fractions, fractions_bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D fractions failed");
        check_cuda(cudaMemset(d_sum_final, 0, aggregate_bytes), "cudaMemset(sum_final) failed");
        check_cuda(cudaMemset(d_sum_log, 0, aggregate_bytes), "cudaMemset(sum_log) failed");
        check_cuda(cudaMemset(d_ruin_count, 0, ruin_bytes), "cudaMemset(ruin_count) failed");

        if (sample_count > 0) {
            check_cuda(cudaMemcpy(d_sample_map, h_sample_map.data(), sample_map_bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D sample_map failed");
            check_cuda(cudaMemset(d_sampled_final, 0, sampled_bytes), "cudaMemset(sampled_final) failed");
        }
        auto t_transfer_end = std::chrono::steady_clock::now();
        transfer_time_ms += std::chrono::duration<double, std::milli>(t_transfer_end - t_transfer_start).count();

        int stream_count = std::max(1, std::min(streams, n_fractions));
        cuda_streams.resize(stream_count, nullptr);
        for (int i = 0; i < stream_count; ++i) {
            check_cuda(cudaStreamCreateWithFlags(&cuda_streams[i], cudaStreamNonBlocking), "cudaStreamCreate failed");
        }
        check_cuda(cudaEventCreateWithFlags(&scenario_ready_event, cudaEventDisableTiming), "cudaEventCreate failed");

        auto t_kernel_start = std::chrono::steady_clock::now();
        launch_generate_scenario_params(
            n_sims,
            params,
            scenario.is_crash,
            scenario.crash_start,
            scenario.down_end,
            scenario.crash_end,
            scenario.mu_down,
            scenario.mu_up,
            scenario.layoff_start,
            scenario.layoff_end,
            scenario.summary_counts,
            scenario.summary_sums,
            cuda_streams[0]
        );
        check_cuda(cudaGetLastError(), "Synthetic scenario generation launch failed");
        check_cuda(cudaEventRecord(scenario_ready_event, cuda_streams[0]), "cudaEventRecord failed");

        int base = n_fractions / stream_count;
        int remainder = n_fractions % stream_count;
        int fraction_start = 0;
        for (int i = 0; i < stream_count; ++i) {
            int count = base + (i < remainder ? 1 : 0);
            if (count <= 0) {
                continue;
            }
            if (i != 0) {
                check_cuda(
                    cudaStreamWaitEvent(cuda_streams[i], scenario_ready_event, 0),
                    "cudaStreamWaitEvent failed"
                );
            }
            launch_simulate_fraction_tile_aggregates_from_params(
                scenario.is_crash,
                scenario.crash_start,
                scenario.down_end,
                scenario.crash_end,
                scenario.mu_down,
                scenario.mu_up,
                scenario.layoff_start,
                scenario.layoff_end,
                d_fractions,
                n_sims,
                fraction_start,
                count,
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
                d_sample_map,
                sample_count,
                d_sum_final,
                d_sum_log,
                d_ruin_count,
                d_sampled_final,
                cuda_streams[i]
            );
            check_cuda(cudaGetLastError(), "Synthetic aggregate kernel launch failed");
            fraction_start += count;
        }

        for (cudaStream_t stream : cuda_streams) {
            check_cuda(cudaStreamSynchronize(stream), "Synthetic kernel execution failed");
        }
        auto t_kernel_end = std::chrono::steady_clock::now();
        kernel_time_ms = std::chrono::duration<double, std::milli>(t_kernel_end - t_kernel_start).count();

        std::vector<unsigned long long> h_ruin_temp(static_cast<std::size_t>(n_fractions), 0ull);
        auto t_copy_back_start = std::chrono::steady_clock::now();
        check_cuda(cudaMemcpy(h_sum_final, d_sum_final, aggregate_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy D2H sum_final failed");
        check_cuda(cudaMemcpy(h_sum_log, d_sum_log, aggregate_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy D2H sum_log failed");
        check_cuda(cudaMemcpy(h_ruin_temp.data(), d_ruin_count, ruin_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy D2H ruin_count failed");
        if (sample_count > 0) {
            check_cuda(cudaMemcpy(h_sampled_final, d_sampled_final, sampled_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy D2H sampled_final failed");
        }
        auto t_copy_back_end = std::chrono::steady_clock::now();
        transfer_time_ms += std::chrono::duration<double, std::milli>(t_copy_back_end - t_copy_back_start).count();

        for (int i = 0; i < n_fractions; ++i) {
            h_ruin_count[i] = static_cast<std::int64_t>(h_ruin_temp[static_cast<std::size_t>(i)]);
        }

        summary = copy_summary_to_host(n_sims, scenario);
    } catch (...) {
        cleanup();
        throw;
    }

    cleanup();
    return py::make_tuple(sum_final, sum_log, ruin_count, sampled_final, summary, kernel_time_ms, transfer_time_ms);
}

py::tuple simulate_fraction_tile_synthetic(
    int n_sims,
    py::array_t<double, py::array::c_style | py::array::forcecast> fractions,
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
    py::dict synthetic_params,
    int streams
) {
    auto fractions_info = fractions.request();
    if (fractions_info.ndim != 1) {
        throw std::runtime_error("fractions must be a 1D array.");
    }
    int n_fractions = static_cast<int>(fractions_info.shape[0]);
    if (n_sims <= 0 || n_fractions <= 0) {
        throw std::runtime_error("Invalid n_sims or fraction count for synthetic full path.");
    }

    SyntheticGenerationParams params = parse_synthetic_params(synthetic_params);
    if (params.horizon <= 0) {
        throw std::runtime_error("synthetic_params.horizon must be > 0.");
    }

    py::array_t<double> final_net_worth({n_fractions, n_sims});
    py::array_t<std::uint8_t> ruined({n_fractions, n_sims});

    auto final_info = final_net_worth.request();
    auto ruined_info = ruined.request();

    const double* h_fractions = static_cast<const double*>(fractions_info.ptr);
    double* h_final = static_cast<double*>(final_info.ptr);
    std::uint8_t* h_ruined = static_cast<std::uint8_t*>(ruined_info.ptr);

    ScenarioDeviceBuffers scenario{};
    double* d_fractions = nullptr;
    double* d_final = nullptr;
    std::uint8_t* d_ruined = nullptr;
    cudaStream_t stream = nullptr;

    std::size_t fractions_bytes = static_cast<std::size_t>(n_fractions) * sizeof(double);
    std::size_t out_double_bytes = static_cast<std::size_t>(n_fractions) * n_sims * sizeof(double);
    std::size_t out_ruined_bytes = static_cast<std::size_t>(n_fractions) * n_sims * sizeof(std::uint8_t);

    auto cleanup = [&]() {
        if (stream != nullptr) {
            cudaStreamDestroy(stream);
        }
        cudaFree(d_fractions);
        cudaFree(d_final);
        cudaFree(d_ruined);
        free_scenario_buffers(scenario);
    };

    double kernel_time_ms = 0.0;
    double transfer_time_ms = 0.0;

    try {
        check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate failed");
        allocate_scenario_buffers(n_sims, false, scenario);
        check_cuda(cudaMalloc(&d_fractions, fractions_bytes), "cudaMalloc(fractions) failed");
        check_cuda(cudaMalloc(&d_final, out_double_bytes), "cudaMalloc(final) failed");
        check_cuda(cudaMalloc(&d_ruined, out_ruined_bytes), "cudaMalloc(ruined) failed");

        auto t_transfer_start = std::chrono::steady_clock::now();
        check_cuda(cudaMemcpyAsync(d_fractions, h_fractions, fractions_bytes, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync H2D fractions failed");
        check_cuda(cudaStreamSynchronize(stream), "fraction copy sync failed");
        auto t_transfer_end = std::chrono::steady_clock::now();
        transfer_time_ms += std::chrono::duration<double, std::milli>(t_transfer_end - t_transfer_start).count();

        auto t_kernel_start = std::chrono::steady_clock::now();
        launch_generate_scenario_params(
            n_sims,
            params,
            scenario.is_crash,
            scenario.crash_start,
            scenario.down_end,
            scenario.crash_end,
            scenario.mu_down,
            scenario.mu_up,
            scenario.layoff_start,
            scenario.layoff_end,
            nullptr,
            nullptr,
            stream
        );
        check_cuda(cudaGetLastError(), "Synthetic scenario generation launch failed");

        launch_simulate_fraction_tile_full_from_params(
            scenario.is_crash,
            scenario.crash_start,
            scenario.down_end,
            scenario.crash_end,
            scenario.mu_down,
            scenario.mu_up,
            scenario.layoff_start,
            scenario.layoff_end,
            d_fractions,
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
            d_final,
            d_ruined,
            stream
        );
        check_cuda(cudaGetLastError(), "Synthetic full kernel launch failed");
        check_cuda(cudaStreamSynchronize(stream), "Synthetic full kernel execution failed");
        auto t_kernel_end = std::chrono::steady_clock::now();
        kernel_time_ms = std::chrono::duration<double, std::milli>(t_kernel_end - t_kernel_start).count();

        auto t_copy_back_start = std::chrono::steady_clock::now();
        check_cuda(cudaMemcpy(h_final, d_final, out_double_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy D2H final failed");
        check_cuda(cudaMemcpy(h_ruined, d_ruined, out_ruined_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy D2H ruined failed");
        auto t_copy_back_end = std::chrono::steady_clock::now();
        transfer_time_ms += std::chrono::duration<double, std::milli>(t_copy_back_end - t_copy_back_start).count();
    } catch (...) {
        cleanup();
        throw;
    }

    cleanup();
    return py::make_tuple(final_net_worth, ruined, kernel_time_ms, transfer_time_ms);
}

}  // namespace

PYBIND11_MODULE(blackswan_cuda, m) {
    m.doc() = "CUDA accelerator for blackswan";
    m.def(
        "simulate_fraction_tile",
        &simulate_fraction_tile,
        py::arg("monthly_returns"),
        py::arg("unemployment_matrix"),
        py::arg("fractions"),
        py::arg("initial_portfolio"),
        py::arg("initial_basis"),
        py::arg("ltcg_tax_rate"),
        py::arg("monthly_expenses"),
        py::arg("monthly_savings"),
        py::arg("retirement_enabled"),
        py::arg("retirement_start_month_from_start"),
        py::arg("retirement_safe_withdrawal_rate_annual"),
        py::arg("retirement_expense_reduction_fraction"),
        py::arg("retirement_dynamic_safe_withdrawal_rate"),
        py::arg("reinvest_enabled"),
        py::arg("reinvest_crash_drawdown_threshold"),
        py::arg("reinvest_recovery_fraction_of_peak"),
        py::arg("reinvest_fraction_of_initial_sale_proceeds"),
        py::arg("reinvest_cash_buffer_months"),
        py::arg("cash_yield_annual")
    );
    m.def(
        "simulate_fraction_tile_aggregates",
        &simulate_fraction_tile_aggregates,
        py::arg("monthly_returns"),
        py::arg("unemployment_matrix"),
        py::arg("fractions"),
        py::arg("initial_portfolio"),
        py::arg("initial_basis"),
        py::arg("ltcg_tax_rate"),
        py::arg("monthly_expenses"),
        py::arg("monthly_savings"),
        py::arg("retirement_enabled"),
        py::arg("retirement_start_month_from_start"),
        py::arg("retirement_safe_withdrawal_rate_annual"),
        py::arg("retirement_expense_reduction_fraction"),
        py::arg("retirement_dynamic_safe_withdrawal_rate"),
        py::arg("reinvest_enabled"),
        py::arg("reinvest_crash_drawdown_threshold"),
        py::arg("reinvest_recovery_fraction_of_peak"),
        py::arg("reinvest_fraction_of_initial_sale_proceeds"),
        py::arg("reinvest_cash_buffer_months"),
        py::arg("cash_yield_annual"),
        py::arg("log_utility_wealth_floor"),
        py::arg("sample_positions") = py::none(),
        py::arg("streams") = 1
    );
    m.def(
        "simulate_fraction_tile_aggregates_synthetic",
        &simulate_fraction_tile_aggregates_synthetic,
        py::arg("n_sims"),
        py::arg("fractions"),
        py::arg("initial_portfolio"),
        py::arg("initial_basis"),
        py::arg("ltcg_tax_rate"),
        py::arg("monthly_expenses"),
        py::arg("monthly_savings"),
        py::arg("retirement_enabled"),
        py::arg("retirement_start_month_from_start"),
        py::arg("retirement_safe_withdrawal_rate_annual"),
        py::arg("retirement_expense_reduction_fraction"),
        py::arg("retirement_dynamic_safe_withdrawal_rate"),
        py::arg("reinvest_enabled"),
        py::arg("reinvest_crash_drawdown_threshold"),
        py::arg("reinvest_recovery_fraction_of_peak"),
        py::arg("reinvest_fraction_of_initial_sale_proceeds"),
        py::arg("reinvest_cash_buffer_months"),
        py::arg("cash_yield_annual"),
        py::arg("log_utility_wealth_floor"),
        py::arg("synthetic_params"),
        py::arg("sample_positions") = py::none(),
        py::arg("streams") = 1
    );
    m.def(
        "simulate_fraction_tile_synthetic",
        &simulate_fraction_tile_synthetic,
        py::arg("n_sims"),
        py::arg("fractions"),
        py::arg("initial_portfolio"),
        py::arg("initial_basis"),
        py::arg("ltcg_tax_rate"),
        py::arg("monthly_expenses"),
        py::arg("monthly_savings"),
        py::arg("retirement_enabled"),
        py::arg("retirement_start_month_from_start"),
        py::arg("retirement_safe_withdrawal_rate_annual"),
        py::arg("retirement_expense_reduction_fraction"),
        py::arg("retirement_dynamic_safe_withdrawal_rate"),
        py::arg("reinvest_enabled"),
        py::arg("reinvest_crash_drawdown_threshold"),
        py::arg("reinvest_recovery_fraction_of_peak"),
        py::arg("reinvest_fraction_of_initial_sale_proceeds"),
        py::arg("reinvest_cash_buffer_months"),
        py::arg("cash_yield_annual"),
        py::arg("synthetic_params"),
        py::arg("streams") = 1
    );
}
