cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(simpleCUBLAS LANGUAGES CXX)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 72 75 80 86 87 89 90 100 101 120)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")  # enable cuda-gdb (expensive)
endif()

# Include directories and libraries
include_directories(../../../Common)

# Source file
# Add target for simpleCUBLAS
add_executable(simpleCUBLAS simpleCUBLAS.cpp)

target_compile_options(simpleCUBLAS PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(simpleCUBLAS PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(simpleCUBLAS PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(simpleCUBLAS PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(simpleCUBLAS PRIVATE
    CUDA::cudart
    CUDA::cublas
)
