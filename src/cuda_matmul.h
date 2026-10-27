#pragma once
// Интерфейс CUDA-части (реализация в cuda_matmul.cu). Подключается из main.cpp только при USE_CUDA.

struct CudaTimes {
    double h2d = 0, kernel = 0, d2h = 0, total = 0;  // секунды (cudaEvent)
    int gridX = 0, gridY = 0;
};

// Информация о первой видеокарте. Возвращает число устройств (0 - нет CUDA-устройства).
int cudaDeviceInfo(char* name, int nameLen, double* memGb, int* smCount);

// C = A * B на GPU. A, B, C - плотные матрицы N x N (построчно, double).
// block - сторона блока потоков (блок block x block), tiled=false - наивное ядро, true - с shared-памятью.
bool multiplyCUDA(const double* A, const double* B, double* C, int N, int block, bool tiled, CudaTimes& times);
