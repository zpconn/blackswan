cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_COMPILER)
    if(EXISTS "/usr/local/cuda/bin/nvcc")
        set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" CACHE FILEPATH "Path to nvcc")
    elseif(EXISTS "/usr/local/cuda-13.1/bin/nvcc")
        set(CMAKE_CUDA_COMPILER "/usr/local/cuda-13.1/bin/nvcc" CACHE FILEPATH "Path to nvcc")
    endif()
endif()

if(POLICY CMP0148)
    cmake_policy(SET CMP0148 NEW)
endif()

set(PYBIND11_FINDPYTHON ON)

project(crash_prep_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(pybind11 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

pybind11_add_module(
    crash_prep_cuda
    src/cuda/bindings.cpp
    src/cuda/sim_kernel.cu
)

target_include_directories(crash_prep_cuda PRIVATE src/cuda)
target_link_libraries(crash_prep_cuda PRIVATE CUDA::cudart)

target_compile_options(
    crash_prep_cuda
    PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

install(
    TARGETS crash_prep_cuda
    LIBRARY DESTINATION .
    RUNTIME DESTINATION .
    ARCHIVE DESTINATION .
)
