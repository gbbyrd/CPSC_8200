#pragma once

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

template <const uint BLOCK_SIZE>
__global__ void coalescing(const float *A, const float *B, float *C, 
                            uint M, uint N, uint K) {
                              
  const int cCol = blockIdx.x * BLOCK_SIZE + (threadIdx.x % BLOCK_SIZE);
  const int cRow = blockIdx.y * BLOCK_SIZE + (threadIdx.x / BLOCK_SIZE);

  // if statement is necessary to make things work under tile quantization
  if (cRow < M && cCol < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[cRow * K + i] * B[i * N + cCol];
    }
    C[cRow * N + cCol] = tmp;
  }
}