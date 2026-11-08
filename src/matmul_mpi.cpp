// Лабораторная работа 3: умножение квадратных матриц по технологии MPI.
//
// Программа из л/р №1 (тот же формат входа/выхода и тот же тройной цикл), распараллеленная по строкам:
//   rank 0 читает A и B  ->  MPI_Bcast(B) всем  ->  MPI_Scatterv(строки A)  ->  каждый процесс считает свои строки C
//   ->  MPI_Gatherv(строки C) на rank 0.
// Число строк на процесс отличается не более чем на одну, поэтому N не обязано делиться на число процессов.
//
// Код намеренно написан на C++98 без сторонних библиотек: на суперкомпьютере «Сергей Королёв» установлены
// старые компиляторы (GCC 4.x, Intel MPI 4 / Open MPI 1.4), а здесь собирается любым mpicxx:
//   mpicxx -O3 -o build/matmul_mpi src/matmul_mpi.cpp
//
// Запуск:
//   mpirun -np P ./build/matmul_mpi data/input_200.txt [mpi|seq] [выходной_файл|none]
//     mpi  (по умолчанию) - параллельная версия на P процессах
//     seq  - последовательная версия (считает только rank 0, запускать с -np 1), база для сравнения
//     выходной файл по умолчанию: data/output_N_parallel_mpi_tP.txt или data/output_N_sequential.txt;
//     none - не записывать матрицу (только время в stdout), чтобы повторные замеры не тратили время на запись.
//
// Время измеряется MPI_Wtime между двумя MPI_Barrier и включает все пересылки (чтение/запись файлов не входят).
// В stdout печатается строка "RESULT ..." для скриптов.
#include <mpi.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#include <unistd.h>

namespace {

double* ptr(std::vector<double>& v) { return v.empty() ? 0 : &v[0]; }
int* iptr(std::vector<int>& v) { return v.empty() ? 0 : &v[0]; }

// Тройной цикл из л/р №1 для rows строк: C[i][j] = sum_k A[i][k] * B[k][j].
void multiplyRows(const double* A, const double* B, double* C, int N, int rows) {
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < N; ++k)
                sum += A[(size_t)i * N + k] * B[(size_t)k * N + j];
            C[(size_t)i * N + j] = sum;
        }
}

bool readMatrices(const char* path, int& N, std::vector<double>& A, std::vector<double>& B) {
    FILE* f = std::fopen(path, "r");
    if (!f) return false;
    if (std::fscanf(f, "%d", &N) != 1 || N <= 0) {
        std::fclose(f);
        return false;
    }
    size_t nn = (size_t)N * N;
    A.resize(nn);
    B.resize(nn);
    for (size_t i = 0; i < nn; ++i)
        if (std::fscanf(f, "%lf", &A[i]) != 1) { std::fclose(f); return false; }
    for (size_t i = 0; i < nn; ++i)
        if (std::fscanf(f, "%lf", &B[i]) != 1) { std::fclose(f); return false; }
    std::fclose(f);
    return true;
}

std::string intToStr(long v) {
    char buf[32];
    std::sprintf(buf, "%ld", v);
    return buf;
}

// Число различных узлов, на которых запущены процессы (по MPI_Get_processor_name).
int countNodes(int rank, int size) {
    char name[MPI_MAX_PROCESSOR_NAME];
    std::memset(name, 0, sizeof(name));
    int len = 0;
    MPI_Get_processor_name(name, &len);
    std::vector<char> all;
    if (rank == 0) all.resize((size_t)size * MPI_MAX_PROCESSOR_NAME);
    MPI_Gather(name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, all.empty() ? 0 : &all[0], MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 0,
               MPI_COMM_WORLD);
    if (rank != 0) return 0;
    std::set<std::string> uniq;
    for (int r = 0; r < size; ++r) uniq.insert(std::string(&all[(size_t)r * MPI_MAX_PROCESSOR_NAME]));
    return (int)uniq.size();
}

// Параллельное умножение. Возвращает полное время (раздача + вычисление + сбор);
// computeSeconds (на rank 0) - максимум по процессам времени чистого вычисления.
double multiplyMPI(std::vector<double>& A, std::vector<double>& B, std::vector<double>& C, int N, int rank, int size,
                   double& computeSeconds) {
    std::vector<int> counts(size), displs(size);
    int base = N / size, rest = N % size, off = 0;
    for (int r = 0; r < size; ++r) {
        int rows = base + (r < rest ? 1 : 0);
        counts[r] = rows * N;
        displs[r] = off * N;
        off += rows;
    }
    int myRows = counts[rank] / N;

    std::vector<double> localA((size_t)myRows * N), localC((size_t)myRows * N, 0.0);
    if (rank != 0) B.assign((size_t)N * N, 0.0);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    MPI_Bcast(ptr(B), N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(ptr(A), iptr(counts), iptr(displs), MPI_DOUBLE, ptr(localA), myRows * N, MPI_DOUBLE, 0,
                 MPI_COMM_WORLD);

    double c0 = MPI_Wtime();
    multiplyRows(ptr(localA), ptr(B), ptr(localC), N, myRows);
    double myCompute = MPI_Wtime() - c0;

    MPI_Gatherv(ptr(localC), myRows * N, MPI_DOUBLE, ptr(C), iptr(counts), iptr(displs), MPI_DOUBLE, 0,
                MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    double total = MPI_Wtime() - t0;

    computeSeconds = 0.0;
    MPI_Reduce(&myCompute, &computeSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    return total;
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0)
            std::fprintf(stderr,
                         "Использование: mpirun -np P %s <input_file> [mpi|seq] [output_file|none]\n"
                         "  mpi - параллельная версия (по умолчанию), seq - последовательная (rank 0)\n",
                         argv[0]);
        MPI_Finalize();
        return 1;
    }
    std::string mode = argc >= 3 ? argv[2] : "mpi";
    if (mode != "mpi" && mode != "seq") {
        if (rank == 0) std::fprintf(stderr, "Неизвестный режим '%s' (допустимо: mpi, seq)\n", mode.c_str());
        MPI_Finalize();
        return 1;
    }

    int N = 0;
    std::vector<double> A, B, C;
    if (rank == 0) {
        if (!readMatrices(argv[1], N, A, B)) {
            std::fprintf(stderr, "Не могу прочитать файл: %s\n", argv[1]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        C.assign((size_t)N * N, 0.0);
    }
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int nodes = countNodes(rank, size);

    std::string strategy;
    double seconds = 0.0, computeSeconds = 0.0;
    if (mode == "seq") {
        strategy = "sequential";
        if (rank == 0) {
            if (size > 1) std::fprintf(stderr, "Внимание: режим seq считает только rank 0, остальные процессы простаивают\n");
            double t0 = MPI_Wtime();
            multiplyRows(ptr(A), ptr(B), ptr(C), N, N);
            seconds = computeSeconds = MPI_Wtime() - t0;
        }
    } else {
        strategy = "parallel_mpi";
        seconds = multiplyMPI(A, B, C, N, rank, size, computeSeconds);
    }

    if (rank == 0) {
        int procs = (mode == "seq") ? 0 : size;   // как в л/р 1-2: 0 = последовательная версия
        std::string outputPath = argc >= 4 ? argv[3]
                                           : (mode == "seq" ? "data/output_" + intToStr(N) + "_sequential.txt"
                                                            : "data/output_" + intToStr(N) + "_parallel_mpi_t" +
                                                                  intToStr(size) + ".txt");
        long cores = sysconf(_SC_NPROCESSORS_ONLN);
        if (outputPath != "none") {
            FILE* out = std::fopen(outputPath.c_str(), "w");
            if (!out) {
                std::fprintf(stderr, "Не могу записать файл: %s (создайте папку data/)\n", outputPath.c_str());
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            std::fprintf(out, "%d\n", N);
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) std::fprintf(out, "%g ", C[(size_t)i * N + j]);
                std::fputc('\n', out);
            }
            std::fprintf(out, "strategy %s\n", strategy.c_str());
            std::fprintf(out, "threads %d\n", procs);   // здесь «threads» = число MPI-процессов
            std::fprintf(out, "cores_available %ld\n", cores);
            std::fprintf(out, "nodes %d\n", nodes);
            std::fprintf(out, "time_seconds %.6f\n", seconds);
            std::fprintf(out, "compute_seconds %.6f\n", computeSeconds);
            std::fprintf(out, "comm_seconds %.6f\n", seconds - computeSeconds);
            std::fclose(out);
        }
        std::printf("N=%d strategy=%s processes=%d nodes=%d time=%.6fs (compute=%.6fs)\n", N, strategy.c_str(), size,
                    nodes, seconds, computeSeconds);
        std::printf("RESULT n=%d strategy=%s procs=%d nodes=%d time=%.6f compute=%.6f\n", N, strategy.c_str(), procs,
                    nodes, seconds, computeSeconds);
        std::fflush(stdout);
    }
    MPI_Finalize();
    return 0;
}
