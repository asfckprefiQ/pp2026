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
