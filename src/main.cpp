#include <chrono>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <omp.h>

#ifdef USE_CUDA
#include "cuda_matmul.h"
#endif

using Matrix = std::vector<std::vector<double>>;

Matrix readMatrix(std::ifstream& in, int N) {
    Matrix M(N, std::vector<double>(N));
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            in >> M[i][j];
    return M;
}

// Умножение строк [rowStart, rowEnd) матрицы A на B (тройной цикл, без библиотек).
void multiplyRange(const Matrix& A, const Matrix& B, Matrix& C, int N, int rowStart, int rowEnd) {
    for (int i = rowStart; i < rowEnd; ++i)
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < N; ++k)
                sum += A[i][k] * B[k][j];
            C[i][j] = sum;
        }
}

void multiplySequential(const Matrix& A, const Matrix& B, Matrix& C, int N) {
    multiplyRange(A, B, C, N, 0, N);
}

void multiplyStdThreads(const Matrix& A, const Matrix& B, Matrix& C, int N, int T) {
    std::vector<std::thread> workers;
    int rowsPerThread = N / T;
    for (int t = 0; t < T; ++t) {
        int rowStart = t * rowsPerThread;
        int rowEnd = (t == T - 1) ? N : rowStart + rowsPerThread;
        workers.emplace_back(multiplyRange, std::cref(A), std::cref(B), std::ref(C), N, rowStart, rowEnd);
    }
    for (auto& w : workers) w.join();
}

// Лаба 2: параллельный внешний цикл по строкам результирующей матрицы (OpenMP).
void multiplyOpenMP(const Matrix& A, const Matrix& B, Matrix& C, int N, int T) {
    omp_set_dynamic(0);        // ровно T потоков, без автоподбора
    omp_set_num_threads(T);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < N; ++k)
                sum += A[i][k] * B[k][j];
            C[i][j] = sum;
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Использование: ./matmul <input_file> [threads] [backend]\n";
        std::cerr << "  threads=0 или не указан -> последовательная версия\n";
        std::cerr << "  backend: std (по умолчанию, std::thread) | omp (OpenMP) | cuda\n";
        std::cerr << "  MPI (лаба 3) - отдельная программа: mpirun -np P ./build/matmul_mpi <input_file> [mpi|seq]\n";
        std::cerr << "  cuda: ./matmul <input_file> <block> cuda [naive|tiled]   (block = сторона блока потоков, по умолчанию tiled)\n";
        return 1;
    }
    std::string backend = (argc >= 4) ? argv[3] : "std";

    if (backend == "cuda") {
#ifdef USE_CUDA
        int block = (argc >= 3) ? std::stoi(argv[2]) : 16;
        std::string kernel = (argc >= 5) ? argv[4] : "tiled";
        if (kernel != "naive" && kernel != "tiled") {
            std::cerr << "Неизвестное ядро '" << kernel << "' (допустимо: naive, tiled)\n";
            return 1;
        }
        if (block <= 0) block = 16;

        char devName[256] = "";
        double memGb = 0;
        int sm = 0;
        if (cudaDeviceInfo(devName, sizeof(devName), &memGb, &sm) == 0) {
            std::cerr << "CUDA-устройство не найдено\n";
            return 1;
        }

        std::ifstream in(argv[1]);
        if (!in) {
            std::cerr << "Не могу открыть файл: " << argv[1] << "\n";
            return 1;
        }
        int N;
        in >> N;
        Matrix A2 = readMatrix(in, N), B2 = readMatrix(in, N);
        std::vector<double> A((size_t)N * N), B((size_t)N * N), C((size_t)N * N, 0.0);
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) {
                A[(size_t)i * N + j] = A2[i][j];
                B[(size_t)i * N + j] = B2[i][j];
            }

        CudaTimes tm;
        if (!multiplyCUDA(A.data(), B.data(), C.data(), N, block, kernel == "tiled", tm)) return 1;

        std::string strategy = "cuda_" + kernel;
        std::string outputPath = "data/output_" + std::to_string(N) + "_" + strategy + "_b" + std::to_string(block) + ".txt";
        std::ofstream out(outputPath);
        if (!out) {
            std::cerr << "Не могу записать файл: " << outputPath << " (создайте папку data/)\n";
            return 1;
        }
        out << N << "\n";
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) out << C[(size_t)i * N + j] << " ";
            out << "\n";
        }
        out << "strategy " << strategy << "\n";
        out << "threads " << block << "\n";   // здесь «threads» = сторона блока
        out << "cores_available " << std::thread::hardware_concurrency() << "\n";
        out << "time_seconds " << tm.total << "\n";        // с копированием H2D и D2H
        out << "kernel_seconds " << tm.kernel << "\n";     // только ядро
        out << "h2d_seconds " << tm.h2d << "\n";
        out << "d2h_seconds " << tm.d2h << "\n";
        out << "grid " << tm.gridX << "x" << tm.gridY << "\n";
        out << "device " << devName << "\n";
        out << "sm_count " << sm << "\n";

        std::cout << "N=" << N << " strategy=" << strategy << " block=" << block << "x" << block
                  << " grid=" << tm.gridX << "x" << tm.gridY << " kernel=" << tm.kernel
                  << "s total=" << tm.total << "s\n";
        return 0;
#else
        std::cerr << "Программа собрана без CUDA (нет nvcc при сборке). Установите CUDA Toolkit и пересоберите.\n";
        return 1;
#endif
    }

    std::ifstream in(argv[1]);
    if (!in) {
        std::cerr << "Не могу открыть файл: " << argv[1] << "\n";
        return 1;
    }

    int T = (argc >= 3) ? std::stoi(argv[2]) : 0;
    if (T > 0 && backend != "std" && backend != "omp") {
        std::cerr << "Неизвестный backend '" << backend << "' (допустимо: std, omp, cuda; MPI - программа matmul_mpi)\n";
        return 1;
    }

    int N;
    in >> N;
    Matrix A = readMatrix(in, N);
    Matrix B = readMatrix(in, N);
    Matrix C(N, std::vector<double>(N, 0.0));

    std::string strategy;
    if (T <= 0) {
        strategy = "sequential";
    } else if (backend == "omp") {
        strategy = "parallel_openmp";
    } else {
        strategy = "parallel_threads";
    }

    // Замеряется только вычисление, без чтения и записи файлов.
    auto start = std::chrono::high_resolution_clock::now();
    if (strategy == "sequential") {
        multiplySequential(A, B, C, N);
    } else if (strategy == "parallel_openmp") {
        multiplyOpenMP(A, B, C, N, T);
    } else {
        multiplyStdThreads(A, B, C, N, T);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double seconds = std::chrono::duration<double>(end - start).count();

    unsigned int coresAvailable = std::thread::hardware_concurrency();

    std::string suffix = (strategy == "sequential") ? "_sequential" : ("_" + strategy + "_t" + std::to_string(T));
    std::string outputPath = "data/output_" + std::to_string(N) + suffix + ".txt";
    std::ofstream out(outputPath);
    if (!out) {
        std::cerr << "Не могу записать файл: " << outputPath << " (создайте папку data/)\n";
        return 1;
    }
    out << N << "\n";
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) out << C[i][j] << " ";
        out << "\n";
    }
    out << "strategy " << strategy << "\n";
    out << "threads " << T << "\n";
    out << "cores_available " << coresAvailable << "\n";
    out << "time_seconds " << seconds << "\n";

    std::cout << "N=" << N << " strategy=" << strategy << " threads=" << T
              << " time=" << seconds << "s\n";
    return 0;
}
