cmake_minimum_required(VERSION 3.18)
project(gae_delta_knn LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find pybind11
find_package(pybind11 REQUIRED)

# KNN extension module
pybind11_add_module(_knn_ext
    src/knn_regressor.cpp
    src/bindings.cpp
)

target_include_directories(_knn_ext PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(_knn_ext PRIVATE
    -O3
    -march=native
)
