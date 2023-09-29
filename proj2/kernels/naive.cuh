#pragma once

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void naive(const float *A, const float *B, float *C, 
                            int M, int N, int K) {
  const uint y = blockIdx.x * blockDim.x + threadIdx.x;
  const uint x = blockIdx.y * blockDim.y + threadIdx.y;

  // if statement is necessary to make things work under tile quantization
  if (x < M && y < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[x * K + i] * B[i * N + y];
    }
    // C = α*(A@B)+β*C
    C[x * N + y] = tmp;
  }
}