#include "kernels/naive.cuh"
#include "kernels/coalescing.cuh"
#include "kernels/basic_tiling.cuh"
#include "kernels/block_tiling_1d.cuh"
#include "kernels/block_tiling_2d.cuh"
#include "kernels/vectorize.cuh"

// Utilities and system includes
#include <assert.h>
#include <helper_string.h>  // helper for shared functions common to CUDA Samples

// Grayson's additions
#include <iostream>
#include <fstream>
using namespace std;
#include "CycleTimer.h"

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

// CUDA and CUBLAS functions
#include <helper_functions.h>
#include <helper_cuda.h>

#ifndef min
#define min(a,b) ((a < b) ? a : b)
#endif
#ifndef max
#define max(a,b) ((a > b) ? a : b)
#endif

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
#define BLOCK_SIZE 32

typedef struct _matrixSize      // Optional Command-line multiplier for matrix sizes
{
    unsigned int uiWA, uiHA, uiWB, uiHB, uiWC, uiHC;
} sMatrixSize;

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set matrix multiply on CPU
//! C = A * B
//! @param C          reference data, computed but preallocated
//! @param A          matrix A as provided to device
//! @param B          matrix B as provided to device
//! @param hA         height of matrix A
//! @param wB         width of matrix B
////////////////////////////////////////////////////////////////////////////////
void
matrixMulCPU(float *C, const float *A, const float *B, unsigned int hA, unsigned int wA, unsigned int wB)
{
    for (unsigned int i = 0; i < hA; ++i)
        for (unsigned int j = 0; j < wB; ++j)
        {
            double sum = 0;

            for (unsigned int k = 0; k < wA; ++k)
            {
                double a = A[i * wA + k];
                double b = B[k * wB + j];
                sum += a * b;
            }

            C[i * wB + j] = (float)sum;
        }
}

// Allocates a matrix with random float entries.
void randomInit(float *data, int size)
{
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (float)RAND_MAX;
}

void printDiff(float *data1, float *data2, int width, int height, int iListLength, float fListTol)
{
    printf("Listing first %d Differences > %.6f...\n", iListLength, fListTol);
    int i,j,k;
    int error_count=0;

    for (j = 0; j < height; j++)
    {
        if (error_count < iListLength)
        {
            printf("\n  Row %d:\n", j);
        }

        for (i = 0; i < width; i++)
        {
            k = j * width + i;
            float fDiff = fabs(data1[k] - data2[k]);

            if (fDiff > fListTol)
            {
                if (error_count < iListLength)
                {
                    printf("    Loc(%d,%d)\tCPU=%.5f\tGPU=%.5f\tDiff=%.6f\n", i, j, data1[k], data2[k], fDiff);
                }

                error_count++;
            }
        }
    }

    printf(" \n  Total Errors = %d\n", error_count);
}

void initializeCUDA(int argc, char **argv, int &devID, int &iSizeMultiple, sMatrixSize &matrix_size)
{
    // By default, we use device 0, otherwise we override the device ID based on what is provided at the command line
    cudaError_t error;
    devID = 0;

    if (checkCmdLineFlag(argc, (const char **)argv, "device"))
    {
        devID = getCmdLineArgumentInt(argc, (const char **)argv, "device");
        error = cudaSetDevice(devID);

        if (error != cudaSuccess)
        {
            printf("cudaSetDevice returned error code %d, line(%d)\n", error, __LINE__);
            exit(EXIT_FAILURE);
        }
    }

    // get number of SMs on this GPU
    error = cudaGetDevice(&devID);

    if (error != cudaSuccess)
    {
        printf("cudaGetDevice returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    
    if (checkCmdLineFlag(argc, (const char **)argv, "sizemult"))
    {
        iSizeMultiple = getCmdLineArgumentInt(argc, (const char **)argv, "sizemult");
    }


    // I commented this out because I wanted to see what would happen at larger
    // matrix sizes
    // iSizeMultiple = min(iSizeMultiple, 10);
    // iSizeMultiple = max(iSizeMultiple, 1);

    cudaDeviceProp deviceProp;

    error = cudaGetDeviceProperties(&deviceProp, devID);

    if (error != cudaSuccess)
    {
        printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);

    // use a larger block size for Fermi and above
    int block_size = (deviceProp.major < 2) ? 16 : 32;

    matrix_size.uiWA = 3 * block_size * iSizeMultiple;
    matrix_size.uiHA = 4 * block_size * iSizeMultiple;
    matrix_size.uiWB = 2 * block_size * iSizeMultiple;
    matrix_size.uiHB = 3 * block_size * iSizeMultiple;
    matrix_size.uiWC = 2 * block_size * iSizeMultiple;
    matrix_size.uiHC = 4 * block_size * iSizeMultiple;

    printf("MatrixA(%u,%u), MatrixB(%u,%u), MatrixC(%u,%u)\n",
           matrix_size.uiHA, matrix_size.uiWA,
           matrix_size.uiHB, matrix_size.uiWB,
           matrix_size.uiHC, matrix_size.uiWC);

    if( matrix_size.uiWA != matrix_size.uiHB ||
        matrix_size.uiHA != matrix_size.uiHC ||
        matrix_size.uiWB != matrix_size.uiWC)
    {
       printf("ERROR: Matrix sizes do not match!\n");
       exit(-1);
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test matrix multiply using CUBLAS
////////////////////////////////////////////////////////////////////////////////
int matrixMultiply(int argc, char **argv, int devID, sMatrixSize &matrix_size)
{
    cudaDeviceProp deviceProp;

    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));

    // use a larger block size for Fermi and above
    // int block_size = (deviceProp.major < 2) ? 16 : 32;
    const int block_size = BLOCK_SIZE;

    // set seed for rand()
    srand(2006);

    // allocate host memory for matrices A and B
    unsigned int size_A = matrix_size.uiWA * matrix_size.uiHA;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float *h_A = (float *)malloc(mem_size_A);
    unsigned int size_B = matrix_size.uiWB * matrix_size.uiHB;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float *h_B = (float *)malloc(mem_size_B);

    // set seed for rand()
    srand(2006);

    // initialize host memory
    randomInit(h_A, size_A);
    randomInit(h_B, size_B);

    // allocate device memory
    float *d_A, *d_B, *d_C;
    unsigned int size_C = matrix_size.uiWC * matrix_size.uiHC;
    unsigned int mem_size_C = sizeof(float) * size_C;

    // allocate host memory for the result
    float *h_C      = (float *) malloc(mem_size_C);
    float *h_CUBLAS = (float *) malloc(mem_size_C);
    float *h_CUSTOM = (float *) malloc(mem_size_C);

    checkCudaErrors(cudaMalloc((void **) &d_A, mem_size_A));
    checkCudaErrors(cudaMalloc((void **) &d_B, mem_size_B));
    checkCudaErrors(cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void **) &d_C, mem_size_C));

    int kernelNum = *argv[2] - '0';

    // define pre-kernel information
    string file_name;
    dim3 threads;
    dim3 grid;
    const uint BM = 64;
    const uint BN = 64;
    const uint BK = 8;
    const uint TM = 8;
    const uint TN = 8;
    const uint BM_2d = 64;
    const uint BN_2d = 64;
    switch (kernelNum) {
        case 0:
            threads.x = block_size; threads.y = block_size;
            grid.x = ceil(matrix_size.uiWC / (float)block_size);
            grid.y = ceil(matrix_size.uiHC / (float)block_size);
            file_name = "/home/gbbyrd/CPSC_8200/proj2/naive_benchmark.csv";
            break;
        case 1:
            threads.x = block_size * block_size;
            grid.x = ceil(matrix_size.uiWC / (float)block_size);
            grid.y = ceil(matrix_size.uiHC / (float)block_size);
            file_name = "/home/gbbyrd/CPSC_8200/proj2/coalescing_benchmark.csv";
            break;
        case 2:
            threads.x = block_size * block_size;
            grid.x = ceil(matrix_size.uiWC / (float)block_size);
            grid.y = ceil(matrix_size.uiHC / (float)block_size);
            file_name = "/home/gbbyrd/CPSC_8200/proj2/basic_tiling_benchmark.csv";
            break;
        case 3:
            grid.x = ceil(matrix_size.uiWC / (float)BN);
            grid.y = ceil(matrix_size.uiHC / (float)BM);
            threads.x = BM * BN / TM;
            file_name = "/home/gbbyrd/CPSC_8200/proj2/block_tiling_1d_benchmark.csv";
            break;
        case 4:
            grid.x = ceil(matrix_size.uiWC / (float)BN);
            grid.y = ceil(matrix_size.uiHC / (float)BM);
            threads.x = (BM * BN) / (TM * TN);
            file_name = "/home/gbbyrd/CPSC_8200/proj2/block_tiling_2d_benchmark.csv";
            break;
        case 5:
            grid.x = ceil(matrix_size.uiWC / (float)BN);
            grid.y = ceil(matrix_size.uiHC / (float)BM);
            threads.x = (BM * BN) / (TM * TN);
            file_name = "/home/gbbyrd/CPSC_8200/proj2/vectorize_benchmark.csv";
            break;
        default:
            throw std::invalid_argument("Unknown kernel number");
    }

    // warmup
    switch (kernelNum) {
        case 0:
            naive<<<grid, threads>>>(d_A, d_B, d_C, matrix_size.uiHC,
                                     matrix_size.uiWC, matrix_size.uiWA);
            break;
        case 1:
            coalescing<block_size><<<grid, threads>>>(d_A, d_B, d_C, 
                                                      matrix_size.uiHC,
                                                      matrix_size.uiWC, 
                                                      matrix_size.uiWA);
            break;
        case 2:
            basic_tiling<block_size><<<grid, threads>>>(d_A, d_B, d_C, 
                                                        matrix_size.uiHC,
                                                        matrix_size.uiWC, 
                                                        matrix_size.uiWA);
            break;
        case 3:
            blockTiling1d<BM, BN, BK, TM><<<grid, threads>>>(d_A, d_B, d_C, 
                                                             matrix_size.uiHC,
                                                             matrix_size.uiWC, 
                                                             matrix_size.uiWA);
            break;
        case 4:
            blockTiling2d<BM_2d, BN_2d, BK, TM, TN><<<grid, threads>>>(d_A, d_B, d_C, 
                                                                 matrix_size.uiHC,
                                                                 matrix_size.uiWC, 
                                                                 matrix_size.uiWA);
            break;
        case 5:
            vectorize<BM_2d, BN_2d, BK, TM, TN><<<grid, threads>>>(d_A, d_B, d_C, 
                                                                 matrix_size.uiHC,
                                                                 matrix_size.uiWC, 
                                                                 matrix_size.uiWA);
            break;
        default:
            throw std::invalid_argument("Unknown kernel number");
    }
    cudaDeviceSynchronize();

    // performance test
    cudaEvent_t start, stop;

    // Allocate CUDA events that we'll use for timing
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    int nIter = 5;

    // create and start timer
    printf("Computing result using CUBLAS...");

    // Record the start event
    checkCudaErrors(cudaEventRecord(start, NULL));

    for (int j = 0; j < nIter; j++)
    {
        switch (kernelNum) {
            case 0:
                naive<<<grid, threads>>>(d_A, d_B, d_C, matrix_size.uiHC,
                                         matrix_size.uiWC, matrix_size.uiWA);
                break;
            case 1:
                coalescing<block_size><<<grid, threads>>>(d_A, d_B, d_C, matrix_size.uiHC,
                                              matrix_size.uiWC, matrix_size.uiWA);
                break;
            case 2:
                basic_tiling<block_size><<<grid, threads>>>(d_A, d_B, d_C, 
                                                            matrix_size.uiHC,
                                                            matrix_size.uiWC, 
                                                            matrix_size.uiWA);
                break;
            case 3:
                blockTiling1d<BM, BN, BK, TM><<<grid, threads>>>(d_A, d_B, d_C, 
                                                                 matrix_size.uiHC,
                                                                 matrix_size.uiWC, 
                                                                 matrix_size.uiWA);
                break;
            case 4:
                blockTiling2d<BM_2d, BN_2d, BK, TM, TN><<<grid, threads>>>(d_A, d_B, d_C, 
                                                                    matrix_size.uiHC,
                                                                    matrix_size.uiWC, 
                                                                    matrix_size.uiWA);
                break;
            case 5:
                vectorize<BM_2d, BN_2d, BK, TM, TN><<<grid, threads>>>(d_A, d_B, d_C, 
                                                                    matrix_size.uiHC,
                                                                    matrix_size.uiWC, 
                                                                    matrix_size.uiWA);
                break;
            default:
                throw std::invalid_argument("Unknown kernel number");
        }
        cudaDeviceSynchronize();
    }

    printf("done.\n");

    // Record the stop event
    checkCudaErrors(cudaEventRecord(stop, NULL));

    // Wait for the stop event to complete
    checkCudaErrors(cudaEventSynchronize(stop));

    float msecTotal = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    // Compute and print the performance
    float msecPerMatrixMul = msecTotal / nIter;
    double flopsPerMatrixMul = 2.0 * (double)matrix_size.uiHC * (double)matrix_size.uiWC * (double)matrix_size.uiHB;
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
    printf(
        "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops\n",
        gigaFlops,
        msecPerMatrixMul,
        flopsPerMatrixMul);

    checkCudaErrors(cudaMemcpy(h_CUSTOM, d_C, mem_size_C, cudaMemcpyDeviceToHost));
    
    // kernels are asynchronous, so we must synchronize once the kernel is
    // finished operating
    cudaDeviceSynchronize();

    // write information to the log file for performance testing
    ofstream MyFile;
    MyFile.open(file_name, ios::app);
    MyFile << gigaFlops << ";" << msecPerMatrixMul << ";" << flopsPerMatrixMul/1e5 <<endl;
    MyFile.close();

    // compute reference solution
    printf("Computing result using host CPU...\n\n");
    float *reference = (float *)malloc(mem_size_C);
    // double startTime = CycleTimer::currentSeconds();
    matrixMulCPU(reference, h_A, h_B, matrix_size.uiHA, matrix_size.uiWA, matrix_size.uiWB);
    // double endTime = CycleTimer::currentSeconds();
    // printf("endtime-startTime: %d - %d\n\n", endTime, startTime);
    // printf("[cpu speed]:\t\t[%.3f] ms\n", endTime-startTime * 1000);
    printf("done.\n");

    // check result (CUSTOM)
    bool resCUSTOM = sdkCompareL2fe(reference, h_CUSTOM, size_C, 1.0e-2f);

    if (resCUSTOM != true)
    {
        printDiff(reference, h_CUSTOM, matrix_size.uiWC, matrix_size.uiHC, 100, 1.0e-2f);
    }

    printf("Comparing CUSTOM Matrix Multiply with CPU results: %s\n", (true == resCUSTOM) ? "PASS" : "FAIL");

    printf("\nNOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.\n");

    // clean up memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(reference);
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();

    if (resCUSTOM == true)
    {
        return EXIT_SUCCESS;    // return value = 1
    }
    else
    {
        return EXIT_FAILURE;     // return value = 0
    }
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    printf("[Matrix Multiply CUBLAS] - Starting...\n");

    int devID = 0, sizeMult = 5;
    sMatrixSize matrix_size;

    initializeCUDA(argc, argv, devID, sizeMult, matrix_size);

    int matrix_result = matrixMultiply(argc, argv, devID, matrix_size);

    return matrix_result;
}
