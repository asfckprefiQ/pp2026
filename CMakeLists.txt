cmake_minimum_required(VERSION 3.16)
project(matmul CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

add_executable(matmul src/main.cpp)

find_package(Threads REQUIRED)
target_link_libraries(matmul PRIVATE Threads::Threads)

if(APPLE)
    execute_process(
        COMMAND brew --prefix libomp
        OUTPUT_VARIABLE LIBOMP_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib")
    include_directories(${LIBOMP_PREFIX}/include)
    link_directories(${LIBOMP_PREFIX}/lib)
endif()

find_package(OpenMP REQUIRED)
target_link_libraries(matmul PRIVATE OpenMP::OpenMP_CXX)

# Лаба 3: MPI - отдельная программа matmul_mpi (C++98, чтобы собиралась и старыми mpicxx на кластере).
# Необязательна: без MPI собираются остальные лабы. На кластере собирается без CMake: mpicxx -O3 src/matmul_mpi.cpp
find_package(MPI COMPONENTS CXX)
if(MPI_CXX_FOUND)
    add_executable(matmul_mpi src/matmul_mpi.cpp)
    target_link_libraries(matmul_mpi PRIVATE MPI::MPI_CXX)
else()
    message(STATUS "MPI не найден: программа matmul_mpi (лаба 3) не будет собрана")
endif()

# Лаба 4: CUDA (необязательна: без nvcc собираются лабы 1-3)
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    # CMake >= 3.24 сам определит архитектуру видеокарты; для старого CMake укажите
    # -DCMAKE_CUDA_ARCHITECTURES=75 (T4) / 86 (RTX 30xx) / 89 (RTX 40xx) и т.п. или оставьте по умолчанию
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT CMAKE_VERSION VERSION_LESS 3.24)
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    target_sources(matmul PRIVATE src/cuda_matmul.cu)
    target_include_directories(matmul PRIVATE src)
    target_compile_definitions(matmul PRIVATE USE_CUDA)
    set_target_properties(matmul PROPERTIES CUDA_STANDARD 17)
else()
    message(STATUS "CUDA не найдена: backend cuda будет недоступен")
endif()
