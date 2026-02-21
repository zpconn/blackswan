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

std::pair<py::array_t<double>, py::array_t<std::uint8_t>> simulate_fraction_tile(
    py::array_t<float, py::array::c_style | py::array::forcecast> monthly_returns,
    py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> unemployment_matrix,
    py::array_t<double, py::array::c_style | py::array::forcecast> fractions,
    double initial_portfolio,
    double initial_basis,
    double ltcg_tax_rate,
    double monthly_expenses,
    double monthly_savings,
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

}  // namespace

PYBIND11_MODULE(crash_prep_cuda, m) {
    m.doc() = "CUDA accelerator for crash_prep_optimizer";
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
        py::arg("cash_yield_annual"),
        py::arg("log_utility_wealth_floor"),
        py::arg("sample_positions") = py::none(),
        py::arg("streams") = 1
    );
}
