# QESN-MABe V2: Quantum Energy State Network for Mouse Behavior Classification
# Author: Francisco Angulo de Lafuente
# License: MIT
# Adapted for existing installations: VS2022, Eigen, CUDA 12

cmake_minimum_required(VERSION 3.20)
project(QESN_MABe_V2 VERSION 2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Use existing installations
set(EIGEN3_INCLUDE_DIR "E:/eigen-3.4.0")
set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0")

# Find Eigen3
find_package(Eigen3 3.4 QUIET)
if(NOT Eigen3_FOUND)
    message(STATUS "Using local Eigen3 installation at ${EIGEN3_INCLUDE_DIR}")
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    set_target_properties(Eigen3::Eigen PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}"
    )
endif()

# Find CUDA (optional for GPU acceleration)
find_package(CUDA QUIET)
if(CUDA_FOUND)
    message(STATUS "CUDA found: ${CUDA_VERSION}")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch=sm_75 -std=c++17)
    add_definitions(-DUSE_CUDA)
else()
    message(STATUS "CUDA not found - building CPU-only version")
endif()

# Find OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found - enabling parallel processing")
else()
    message(WARNING "OpenMP not found - will use single-threaded execution")
endif()

# Source files
set(SOURCES
    src/core/quantum_neuron.cpp
    src/core/quantum_foam.cpp
    src/io/dataset_loader.cpp
    src/training/trainer.cpp
    src/main.cpp
)

# Executable
add_executable(qesn_train ${SOURCES})

target_include_directories(qesn_train PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
)

# Link Eigen3
target_link_libraries(qesn_train PRIVATE
    Eigen3::Eigen
)

# Link OpenMP if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(qesn_train PRIVATE OpenMP::OpenMP_CXX)
endif()

# Link CUDA if available
if(CUDA_FOUND)
    target_include_directories(qesn_train PRIVATE ${CUDA_INCLUDE_DIRS})
    target_link_libraries(qesn_train PRIVATE ${CUDA_LIBRARIES})
endif()

# Compiler warnings
if(MSVC)
    target_compile_options(qesn_train PRIVATE /W4 /bigobj)
    # Disable specific warnings for Apache Arrow compatibility
    target_compile_options(qesn_train PRIVATE /wd4251 /wd4275)
else()
    target_compile_options(qesn_train PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Optimization flags for Release
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(MSVC)
        target_compile_options(qesn_train PRIVATE /O2 /Ob2 /Oi /Ot /GL)
        set_target_properties(qesn_train PROPERTIES LINK_FLAGS "/LTCG")
    else()
        target_compile_options(qesn_train PRIVATE -O3 -march=native)
    endif()
endif()

# Installation
install(TARGETS qesn_train DESTINATION bin)

message(STATUS "")
message(STATUS "========================================")
message(STATUS "QESN-MABe V2 Configuration Summary")
message(STATUS "========================================")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "Eigen3: ${EIGEN3_INCLUDE_DIR}")
if(CUDA_FOUND)
    message(STATUS "CUDA: ${CUDA_VERSION} (GPU acceleration enabled)")
else()
    message(STATUS "CUDA: Not found (CPU-only build)")
endif()
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP: Enabled")
else()
    message(STATUS "OpenMP: Disabled")
endif()
message(STATUS "========================================")
message(STATUS "")

# Note about Apache Arrow/Parquet
message(STATUS "NOTE: Apache Arrow/Parquet reading will be handled via Python preprocessing")
message(STATUS "      Run: python scripts/preprocess_parquet.py before training")
message(STATUS "")
