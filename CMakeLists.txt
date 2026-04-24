# QESN-MABe V2: Quantum-inspired Echo State Network (classical energy diffusion)
# Author: Francisco Angulo de Lafuente
# License: Apache-2.0
#
# Portable CMake. No hardcoded absolute paths. CUDA and Arrow are OFF by default.

cmake_minimum_required(VERSION 3.20)
project(QESN_MABe_V2 VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(QESN_WITH_CUDA  "Enable CUDA GPU acceleration"         OFF)
option(QESN_WITH_ARROW "Enable Apache Arrow/Parquet support"  OFF)
option(QESN_WITH_OPENMP "Enable OpenMP parallelism"           ON)

# ---------------------------------------------------------------------------
# Eigen3: try find_package, fall back to FetchContent.
# ---------------------------------------------------------------------------
find_package(Eigen3 3.4 QUIET CONFIG)
if(NOT Eigen3_FOUND)
    message(STATUS "Eigen3 not found via find_package; falling back to FetchContent.")
    include(FetchContent)
    FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG        3.4.0
    )
    # Eigen's CMake pulls in a lot of extras; disable the ones we don't need.
    set(EIGEN_BUILD_DOC     OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING       OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(eigen)
endif()

# ---------------------------------------------------------------------------
# Optional: OpenMP
# ---------------------------------------------------------------------------
if(QESN_WITH_OPENMP)
    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP: enabled")
    else()
        message(STATUS "OpenMP: requested but not found - continuing single-threaded")
    endif()
endif()

# ---------------------------------------------------------------------------
# Optional: CUDA
# ---------------------------------------------------------------------------
if(QESN_WITH_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        message(STATUS "CUDA: enabled (${CMAKE_CUDA_COMPILER})")
    else()
        message(STATUS "CUDA: requested but no CUDA compiler found - building CPU-only")
        set(QESN_WITH_CUDA OFF)
    endif()
endif()

# ---------------------------------------------------------------------------
# Optional: Apache Arrow / Parquet
# ---------------------------------------------------------------------------
if(QESN_WITH_ARROW)
    find_package(Arrow QUIET CONFIG)
    find_package(Parquet QUIET CONFIG)
    if(NOT Arrow_FOUND OR NOT Parquet_FOUND)
        message(STATUS "Arrow/Parquet requested but not found - disabling.")
        set(QESN_WITH_ARROW OFF)
    else()
        message(STATUS "Arrow: ${Arrow_VERSION}  Parquet: ${Parquet_VERSION}")
    endif()
endif()

# ---------------------------------------------------------------------------
# Sources (kept only if they exist on disk, so a partial checkout still configures).
# ---------------------------------------------------------------------------
set(_candidate_sources
    src/core/quantum_neuron.cpp
    src/core/quantum_foam.cpp
    src/io/dataset_loader.cpp
    src/training/trainer.cpp
    src/main.cpp
)
set(SOURCES "")
foreach(_f IN LISTS _candidate_sources)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_f}")
        list(APPEND SOURCES "${_f}")
    endif()
endforeach()

if(SOURCES)
    add_executable(qesn_train ${SOURCES})
    target_include_directories(qesn_train PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(qesn_train PRIVATE Eigen3::Eigen)

    if(OpenMP_CXX_FOUND)
        target_link_libraries(qesn_train PRIVATE OpenMP::OpenMP_CXX)
    endif()

    if(QESN_WITH_CUDA)
        target_compile_definitions(qesn_train PRIVATE QESN_USE_CUDA=1)
    endif()

    if(QESN_WITH_ARROW)
        target_link_libraries(qesn_train PRIVATE Arrow::arrow_shared Parquet::parquet_shared)
        target_compile_definitions(qesn_train PRIVATE QESN_USE_ARROW=1)
    endif()

    if(MSVC)
        target_compile_options(qesn_train PRIVATE /W4 /bigobj /wd4251 /wd4275)
    else()
        target_compile_options(qesn_train PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    install(TARGETS qesn_train RUNTIME DESTINATION bin)
else()
    message(STATUS "No C++ sources found on disk - configuring Python-only package.")
endif()

message(STATUS "")
message(STATUS "========================================")
message(STATUS "QESN-MABe V2 Configuration Summary")
message(STATUS "========================================")
message(STATUS "C++ Standard : ${CMAKE_CXX_STANDARD}")
message(STATUS "Build Type   : ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler     : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "CUDA         : ${QESN_WITH_CUDA}")
message(STATUS "Arrow        : ${QESN_WITH_ARROW}")
message(STATUS "OpenMP       : ${QESN_WITH_OPENMP}")
message(STATUS "Sources found: ${SOURCES}")
message(STATUS "========================================")
