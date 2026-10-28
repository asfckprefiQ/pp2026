#include <stdio.h>
#include <algorithm>
#include <getopt.h>
#include <cstdlib>
#include <cstring>
#include <string>

#include "CycleTimer.h"

extern void mandelbrotSerial(
    float x0, float y0, float x1, float y1,
    int width, int height,
    int startRow, int numRows,
    int maxIterations,
    int output[]);

extern void mandelbrotThread(
    int numThreads,
    float x0, float y0, float x1, float y1,
    int width, int height,
    int maxIterations,
    int output[]);

// Лаба 5: способ разбиения работы между потоками (0 - блоки строк, 1 - чередование строк)
// и время работы каждого потока в последнем вызове mandelbrotThread() (см. mandelbrotThread.cpp).
extern int g_decomposition;
extern double g_threadSeconds[];

extern void writePPMImage(
    int* data,
    int width, int height,
    const char *filename,
    int maxIterations);

void
scaleAndShift(float& x0, float& x1, float& y0, float& y1,
              float scale,
              float shiftX, float shiftY)
{

    x0 *= scale;
    x1 *= scale;
    y0 *= scale;
    y1 *= scale;
    x0 += shiftX;
    x1 += shiftX;
    y0 += shiftY;
    y1 += shiftY;

}

void usage(const char* progname) {
    printf("Usage: %s [options]\n", progname);
    printf("Program Options:\n");
    printf("  -t  --threads <N>  Use N threads\n");
    printf("  -v  --view <INT>   Use specified view settings\n");
    printf("  -m  --mode <M>     Work decomposition: block | interleaved (default: interleaved)\n");
    printf("  -T  --thread-times Print the time of each thread (best run)\n");
    printf("  -c  --csv          Print one machine-readable CSV line (used by tools/run_experiments_lab5.sh)\n");
    printf("  -?  --help         This message\n");
}

bool verifyResult (int *gold, int *result, int width, int height) {

    int i, j;

    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            if (gold[i * width + j] != result[i * width + j]) {
                printf ("Mismatch : [%d][%d], Expected : %d, Actual : %d\n",
                            i, j, gold[i * width + j], result[i * width + j]);
                return 0;
            }
        }
    }

    return 1;
}

int main(int argc, char** argv) {

    const unsigned int width = 1600;
    const unsigned int height = 1200;
    const int maxIterations = 256;
    int numThreads = 2;
    int view = 1;
    bool showThreadTimes = false;
    bool csv = false;

    float x0 = -2;
    float x1 = 1;
    float y0 = -1;
    float y1 = 1;

    // parse commandline options ////////////////////////////////////////////
    int opt;
    static struct option long_options[] = {
        {"threads", 1, 0, 't'},
        {"view", 1, 0, 'v'},
        {"mode", 1, 0, 'm'},
        {"thread-times", 0, 0, 'T'},
        {"csv", 0, 0, 'c'},
        {"help", 0, 0, '?'},
        {0 ,0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "t:v:m:Tc?", long_options, NULL)) != EOF) {

        switch (opt) {
        case 't':
        {
            numThreads = atoi(optarg);
            break;
        }
        case 'v':
        {
            int viewIndex = atoi(optarg);
            view = viewIndex;
            // change view settings
            if (viewIndex == 2) {
                float scaleValue = .015f;
                float shiftX = -.986f;
                float shiftY = .30f;
                scaleAndShift(x0, x1, y0, y1, scaleValue, shiftX, shiftY);
            } else if (viewIndex > 1) {
                fprintf(stderr, "Invalid view index\n");
                return 1;
            }
            break;
        }
        case 'm':
        {
            std::string m = optarg;
            if (m == "block") {
                g_decomposition = 0;
            } else if (m == "interleaved") {
                g_decomposition = 1;
            } else {
                fprintf(stderr, "Invalid mode '%s' (use block or interleaved)\n", optarg);
                return 1;
            }
            break;
        }
        case 'T':
            showThreadTimes = true;
            break;
        case 'c':
            csv = true;
            break;
        case '?':
        default:
            usage(argv[0]);
            return 1;
        }
    }
    // end parsing of commandline options


    int* output_serial = new int[width*height];
    int* output_thread = new int[width*height];
    
    //
    // Run the serial implementation.  Run the code three times and
    // take the minimum to get a good estimate.
    //

    double minSerial = 1e30;
    for (int i = 0; i < 5; ++i) {
       memset(output_serial, 0, width * height * sizeof(int));
        double startTime = CycleTimer::currentSeconds();
        mandelbrotSerial(x0, y0, x1, y1, width, height, 0, height, maxIterations, output_serial);
        double endTime = CycleTimer::currentSeconds();
        minSerial = std::min(minSerial, endTime - startTime);
    }

    printf("[mandelbrot serial]:\t\t[%.3f] ms\n", minSerial * 1000);
    writePPMImage(output_serial, width, height, "mandelbrot-serial.ppm", maxIterations);

    //
    // Run the threaded version
    //

    double minThread = 1e30;
    double bestThreadTimes[32] = {0};   // время каждого потока в самом быстром прогоне
    for (int i = 0; i < 5; ++i) {
      memset(output_thread, 0, width * height * sizeof(int));
        double startTime = CycleTimer::currentSeconds();
        mandelbrotThread(numThreads, x0, y0, x1, y1, width, height, maxIterations, output_thread);
        double endTime = CycleTimer::currentSeconds();
        if (endTime - startTime < minThread) {
            minThread = endTime - startTime;
            memcpy(bestThreadTimes, g_threadSeconds, sizeof(bestThreadTimes));
        }
    }

    printf("[mandelbrot thread]:\t\t[%.3f] ms\n", minThread * 1000);
    writePPMImage(output_thread, width, height, "mandelbrot-thread.ppm", maxIterations);

    if (! verifyResult (output_serial, output_thread, width, height)) {
        printf ("Error : Output from threads does not match serial output\n");

        delete[] output_serial;
        delete[] output_thread;

        return 1;
    }

    // compute speedup
    printf("\t\t\t\t(%.2fx speedup from %d threads)\n", minSerial/minThread, numThreads);

    if (showThreadTimes) {
        for (int t = 0; t < numThreads; t++)
            printf("  thread %d: %.3f ms\n", t, bestThreadTimes[t] * 1000);
    }
    if (csv) {
        // CSV,view,mode,threads,serial_ms,thread_ms,speedup,t0;t1;...  (время потоков в мс)
        printf("CSV,%d,%s,%d,%.3f,%.3f,%.4f,", view, g_decomposition == 0 ? "block" : "interleaved",
               numThreads, minSerial * 1000, minThread * 1000, minSerial / minThread);
        for (int t = 0; t < numThreads; t++)
            printf(t ? ";%.3f" : "%.3f", bestThreadTimes[t] * 1000);
        printf("\n");
    }

    delete[] output_serial;
    delete[] output_thread;

    return 0;
}
