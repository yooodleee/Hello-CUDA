cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(simpleAssert_nvrtc LANGUAGES C CXX CUDA)

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
# Add sample target executable
add_executable(simpleAssert_nvrtc simpleAssert.cpp)

target_compile_options(simpleAssert_nvrtc PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(simpleAssert_nvrtc PRIVATE cxx_std_17 cuda_std_17)

target_link_libraries(simpleAssert_nvrtc PRIVATE
    CUDA::nvrtc
    CUDA::cuda_driver
)

# Copy clock_kernel.cu to the output directory
add_custom_command(TARGET simpleAssert_nvrtc POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/simpleAssert_kernel.cu ${CMAKE_CURRENT_BINARY_DIR}
)
