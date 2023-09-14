#include <stdio.h>
#include <pthread.h>
#include <algorithm>
#include <iostream>
#include <typeinfo>
#include "CycleTimer.h"

using namespace std;

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

static inline int mandel(float c_re, float c_im, int count)
{
    float z_re = c_re, z_im = c_im;
    int i;
    for (i = 0; i < count; ++i) {

        if (z_re * z_re + z_im * z_im > 4.f)
            break;

        float new_re = z_re*z_re - z_im*z_im;
        float new_im = 2.f * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }

    return i;
}

void mandelbrotSpeedup(
    float x0, float y0, float x1, float y1,
    int width, int height,
    int startRow, int stride,
    int maxIterations,
    int output[])
{
    float dx = (x1 - x0) / width;
    float dy = (y1 - y0) / height;

    int row = startRow;
    while (row < height) {
        for (int i = 0; i < width; ++i) {
            float x = x0 + i * dx;
            float y = y0 + row * dy;

            int index = (row * width + i);
            output[index] = mandel(x, y, maxIterations);
        }
        row += stride;
    }
}

//
// workerThreadStart --
//
// Thread entrypoint.
void* workerThreadStart(void* threadArgs) {
    
    
    WorkerArgs* args = static_cast<WorkerArgs*>(threadArgs);
    // calculate the start and end row based on the number of threads and
    // thread id
    int rowsPerThread = args->height / args->numThreads;
    int startingRow = rowsPerThread * args->threadId;
    int totalRows = std::min(rowsPerThread, static_cast<int>(args->height)-startingRow);

    // printf("Total rows: %d Start rows: %d ID: %d\n", totalRows, startingRow, args->threadId);
    double startTime = CycleTimer::currentSeconds();
    mandelbrotSerial(args->x0, args->y0, args->x1, args->y1,
        args->width, args->height, startingRow, totalRows, args->maxIterations,
        args->output);
    double endTime = CycleTimer::currentSeconds();
    printf("Thread %d finished in %f milliseconds\n", args->threadId, (endTime-startTime) * 1000);

    return NULL;
}

void* workerThreadStartSpeedup(void* threadArgs) {
    
    
    WorkerArgs* args = static_cast<WorkerArgs*>(threadArgs);
    // calculate the start and end row based on the number of threads and
    // thread id
    int stride = args->numThreads;
    int startingRow = args->threadId;

    double startTime = CycleTimer::currentSeconds();
    mandelbrotSpeedup(args->x0, args->y0, args->x1, args->y1,
        args->width, args->height, startingRow, stride, args->maxIterations,
        args->output);
    double endTime = CycleTimer::currentSeconds();
    printf("Thread %d finished in %f milliseconds\n", args->threadId, (endTime-startTime) * 1000);

    return NULL;
}

//
// MandelbrotThread --
//
// Multi-threaded implementation of mandelbrot set image generation.
// Multi-threading performed via pthreads.
void mandelbrotThread(
    int numThreads,
    float x0, float y0, float x1, float y1,
    int width, int height,
    int maxIterations, int output[])
{
    const static int MAX_THREADS = 32;

    if (numThreads > MAX_THREADS)
    {
        fprintf(stderr, "Error: Max allowed threads is %d\n", MAX_THREADS);
        exit(1);
    }

    pthread_t workers[MAX_THREADS];
    WorkerArgs args[MAX_THREADS];

    for (int i=0; i<numThreads; i++) {
        args[i].x0 = x0; args[i].y0 = y0;
        args[i].x1 = x1; args[i].y1 = y1;
        args[i].width = width;
        args[i].height = height;
        args[i].maxIterations = maxIterations;
        args[i].output = output;
        args[i].numThreads = numThreads;
        args[i].threadId = i;
    }

    // Fire up the worker threads.  Note that numThreads-1 pthreads
    // are created and the main app thread is used as a worker as
    // well.

    for (int i=1; i<numThreads; i++)
        pthread_create(&workers[i], NULL, workerThreadStart, &args[i]);

    workerThreadStart(&args[0]);

    // wait for worker threads to complete
    for (int i=1; i<numThreads; i++)
        pthread_join(workers[i], NULL);
}

void mandelbrotThreadSpeedup(
    int numThreads,
    float x0, float y0, float x1, float y1,
    int width, int height,
    int maxIterations, int output[])
{
    const static int MAX_THREADS = 32;

    if (numThreads > MAX_THREADS)
    {
        fprintf(stderr, "Error: Max allowed threads is %d\n", MAX_THREADS);
        exit(1);
    }

    pthread_t workers[MAX_THREADS];
    WorkerArgs args[MAX_THREADS];

    for (int i=0; i<numThreads; i++) {
        args[i].x0 = x0; args[i].y0 = y0;
        args[i].x1 = x1; args[i].y1 = y1;
        args[i].width = width;
        args[i].height = height;
        args[i].maxIterations = maxIterations;
        args[i].output = output;
        args[i].numThreads = numThreads;
        args[i].threadId = i;
    }

    // Fire up the worker threads.  Note that numThreads-1 pthreads
    // are created and the main app thread is used as a worker as
    // well.

    for (int i=1; i<numThreads; i++)
        pthread_create(&workers[i], NULL, workerThreadStartSpeedup, &args[i]);

    workerThreadStartSpeedup(&args[0]);

    // wait for worker threads to complete
    for (int i=1; i<numThreads; i++)
        pthread_join(workers[i], NULL);
}
