#include <stdio.h>
#include <stdlib.h>
#include <thread>

#include "CycleTimer.h"

typedef struct {
    float x0, x1;
    float y0, y1;
    unsigned int width;
    unsigned int height;
    int maxIterations;
    int* output;
    int threadId;
    int numThreads;
} WorkerArgs;


extern void mandelbrotSerial(
    float x0, float y0, float x1, float y1,
    int width, int height,
    int startRow, int numRows,
    int maxIterations,
    int output[]);

// Способ разбиения работы между потоками (выбирается ключом -m/--mode в main.cpp):
//   0 - блоки: поток i получает один непрерывный блок строк (пункты 1-3 задания, пространственная декомпозиция);
//   1 - чередование: поток i считает строки i, i+T, i+2T, ... (пункт 4 задания, улучшенное разбиение).
int g_decomposition = 1;

// Время работы каждого потока (секунды) в последнем вызове mandelbrotThread(); поток i пишет только в свою ячейку,
// поэтому синхронизация не нужна (пункт 3 задания).
double g_threadSeconds[32] = {0};

//
// workerThreadStart --
//
// Thread entrypoint.
void workerThreadStart(WorkerArgs * const args) {

    double startTime = CycleTimer::currentSeconds();

    if (g_decomposition == 0) {
        // Блоки: границы считаются целочисленно, поэтому высота не обязана делиться на число потоков
        // (разница в числе строк между потоками не больше одной).
        int startRow = (int)((long long)args->threadId * args->height / args->numThreads);
        int endRow = (int)((long long)(args->threadId + 1) * args->height / args->numThreads);
        mandelbrotSerial(args->x0, args->y0, args->x1, args->y1,
                         args->width, args->height,
                         startRow, endRow - startRow,
                         args->maxIterations, args->output);
    } else {
        // Чередование строк (статическое циклическое распределение): дорогие и дешёвые строки
        // равномерно перемешиваются между потоками, синхронизация между потоками не нужна.
        for (int row = args->threadId; row < (int)args->height; row += args->numThreads) {
            mandelbrotSerial(args->x0, args->y0, args->x1, args->y1,
                             args->width, args->height,
                             row, 1,
                             args->maxIterations, args->output);
        }
    }

    g_threadSeconds[args->threadId] = CycleTimer::currentSeconds() - startTime;
}

//
// MandelbrotThread --
//
// Multi-threaded implementation of mandelbrot set image generation.
// Threads of execution are created by spawning std::threads.
void mandelbrotThread(
    int numThreads,
    float x0, float y0, float x1, float y1,
    int width, int height,
    int maxIterations, int output[])
{
    static constexpr int MAX_THREADS = 32;

    if (numThreads > MAX_THREADS)
    {
        fprintf(stderr, "Error: Max allowed threads is %d\n", MAX_THREADS);
        exit(1);
    }

    // Creates thread objects that do not yet represent a thread.
    std::thread workers[MAX_THREADS];
    WorkerArgs args[MAX_THREADS];

    for (int i=0; i<numThreads; i++) {

        // Всем потокам передаются одинаковые параметры, различаются только threadId;
        // нужная часть изображения определяется в workerThreadStart().
        args[i].x0 = x0;
        args[i].y0 = y0;
        args[i].x1 = x1;
        args[i].y1 = y1;
        args[i].width = width;
        args[i].height = height;
        args[i].maxIterations = maxIterations;
        args[i].numThreads = numThreads;
        args[i].output = output;

        args[i].threadId = i;
    }

    // Spawn the worker threads.  Note that only numThreads-1 std::threads
    // are created and the main application thread is used as a worker
    // as well.
    for (int i=1; i<numThreads; i++) {
        workers[i] = std::thread(workerThreadStart, &args[i]);
    }

    workerThreadStart(&args[0]);

    // join worker threads
    for (int i=1; i<numThreads; i++) {
        workers[i].join();
    }
}
