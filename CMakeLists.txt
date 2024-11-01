

cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(FasterLlama LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 75)

set(COMMON_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

set(COMMON_LIB_DIRS
    /usr/local/cuda-11.7/lib64
)

include_directories(
  ${COMMON_HEADER_DIRS}
)

message("-- Assign include directories (include_directories=${COMMON_HEADER_DIRS})")


add_subdirectory(fasterLlama)
add_subdirectory(samples)