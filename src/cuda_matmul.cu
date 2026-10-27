// Лаба 4 (CUDA): умножение квадратных матриц C = A * B на GPU.
// Два ядра:
//   naive - каждый поток считает один элемент C(row, col), данные читаются из глобальной памяти;
//   tiled - блок загружает плитки A и B в разделяемую (shared) память, глобальных обращений меньше.
// Конфигурация сетки: блок BLOCK x BLOCK потоков, сетка ceil(N/BLOCK) x ceil(N/BLOCK) блоков.
// Тип данных double (как в остальных лабах), поэтому результат сверяется с NumPy с тем же допуском.

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <vector>

#include "cuda_matmul.h"

#define CUDA_OK(call)                                                                         \
    do {                                                                                      \
        cudaError_t err__ = (call);                                                           \
        if (err__ != cudaSuccess) {                                                           \
            std::fprintf(stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString(err__), __FILE__, __LINE__); \
            return false;                                                                     \
        }                                                                                     \
    } while (0)

// --- ядро 1: наивное ---------------------------------------------------------------------
__global__ void matmulNaive(const double* A, const double* B, double* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        double sum = 0.0;
        for (int k = 0; k < N; ++k)
            sum += A[(size_t)row * N + k] * B[(size_t)k * N + col];
        C[(size_t)row * N + col] = sum;
    }
}

// --- ядро 2: плитки в shared-памяти (размер плитки = blockDim.x = blockDim.y) -------------------
__global__ void matmulTiled(const double* A, const double* B, double* C, int N) {
    extern __shared__ double shm[];
    const int T = blockDim.x;
    double* tileA = shm;
    double* tileB = shm + T * T;

    int row = blockIdx.y * T + threadIdx.y;
    int col = blockIdx.x * T + threadIdx.x;
    double sum = 0.0;

    for (int t = 0; t < (N + T - 1) / T; ++t) {
        int aCol = t * T + threadIdx.x;
        int bRow = t * T + threadIdx.y;
        tileA[threadIdx.y * T + threadIdx.x] = (row < N && aCol < N) ? A[(size_t)row * N + aCol] : 0.0;
        tileB[threadIdx.y * T + threadIdx.x] = (bRow < N && col < N) ? B[(size_t)bRow * N + col] : 0.0;
        __syncthreads();
        for (int k = 0; k < T; ++k)
            sum += tileA[threadIdx.y * T + k] * tileB[k * T + threadIdx.x];
        __syncthreads();
    }
    if (row < N && col < N)
        C[(size_t)row * N + col] = sum;
}

int cudaDeviceInfo(char* name, int nameLen, double* memGb, int* smCount) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 0;
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) return 0;
    std::snprintf(name, nameLen, "%s (compute capability %d.%d)", p.name, p.major, p.minor);
    *memGb = p.totalGlobalMem / (1024.0 * 1024.0 * 1024.0);
    *smCount = p.multiProcessorCount;
    return count;
}

bool multiplyCUDA(const double* A, const double* B, double* C, int N, int block, bool tiled,
                  CudaTimes& times) {
    if (block < 1 || block * block > 1024) {
        std::fprintf(stderr, "Размер блока %d недопустим (нужно 1 <= block, block*block <= 1024)\n", block);
        return false;
    }
    size_t bytes = (size_t)N * N * sizeof(double);
    double *dA = nullptr, *dB = nullptr, *dC = nullptr;

    CUDA_OK(cudaFree(0));  // инициализация контекста вне замеров
    CUDA_OK(cudaMalloc(&dA, bytes));
    CUDA_OK(cudaMalloc(&dB, bytes));
    CUDA_OK(cudaMalloc(&dC, bytes));

    cudaEvent_t e0, e1, e2, e3;
    CUDA_OK(cudaEventCreate(&e0)); CUDA_OK(cudaEventCreate(&e1));
    CUDA_OK(cudaEventCreate(&e2)); CUDA_OK(cudaEventCreate(&e3));

    dim3 threads(block, block);
    dim3 grid((N + block - 1) / block, (N + block - 1) / block);
    size_t shmem = tiled ? 2 * (size_t)block * block * sizeof(double) : 0;

    // прогревочный запуск (не входит в замеры), чтобы не считать загрузку модуля ядра
    if (tiled) matmulTiled<<<grid, threads, shmem>>>(dA, dB, dC, N);
    else       matmulNaive<<<grid, threads>>>(dA, dB, dC, N);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());

    CUDA_OK(cudaEventRecord(e0));
    CUDA_OK(cudaMemcpy(dA, A, bytes, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dB, B, bytes, cudaMemcpyHostToDevice));
    CUDA_OK(cudaEventRecord(e1));
    if (tiled) matmulTiled<<<grid, threads, shmem>>>(dA, dB, dC, N);
    else       matmulNaive<<<grid, threads>>>(dA, dB, dC, N);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaEventRecord(e2));
    CUDA_OK(cudaMemcpy(C, dC, bytes, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaEventRecord(e3));
    CUDA_OK(cudaEventSynchronize(e3));

    float msH2D = 0, msKernel = 0, msD2H = 0, msTotal = 0;
    CUDA_OK(cudaEventElapsedTime(&msH2D, e0, e1));
    CUDA_OK(cudaEventElapsedTime(&msKernel, e1, e2));
    CUDA_OK(cudaEventElapsedTime(&msD2H, e2, e3));
    CUDA_OK(cudaEventElapsedTime(&msTotal, e0, e3));
    times.h2d = msH2D / 1000.0;
    times.kernel = msKernel / 1000.0;
    times.d2h = msD2H / 1000.0;
    times.total = msTotal / 1000.0;
    times.gridX = grid.x;
    times.gridY = grid.y;

    cudaEventDestroy(e0); cudaEventDestroy(e1); cudaEventDestroy(e2); cudaEventDestroy(e3);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return true;
}
