#include <chrono>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

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

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Использование: ./matmul <input_file> [threads] [backend]\n";
        std::cerr << "  threads=0 или не указан -> последовательная версия\n";
        std::cerr << "  backend: std (по умолчанию, std::thread)\n";
        return 1;
    }
    std::ifstream in(argv[1]);
    if (!in) {
        std::cerr << "Не могу открыть файл: " << argv[1] << "\n";
        return 1;
    }

    int T = (argc >= 3) ? std::stoi(argv[2]) : 0;
    std::string backend = (argc >= 4) ? argv[3] : "std";
    if (T > 0 && backend != "std") {
        std::cerr << "Backend '" << backend << "' недоступен в этой версии (есть только std)\n";
        return 1;
    }

    int N;
    in >> N;
    Matrix A = readMatrix(in, N);
    Matrix B = readMatrix(in, N);
    Matrix C(N, std::vector<double>(N, 0.0));

    std::string strategy = (T <= 0) ? "sequential" : "parallel_threads";

    // Замеряется только вычисление, без чтения и записи файлов.
    auto start = std::chrono::high_resolution_clock::now();
    if (strategy == "sequential") {
        multiplySequential(A, B, C, N);
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
