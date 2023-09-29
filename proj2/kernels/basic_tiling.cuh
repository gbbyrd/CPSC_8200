#pragma once

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BLOCK_SIZE>
__global__ void basic_tiling(const float *A, const float *B, float *C, 
                            int M, int N, int K) {
  // the output block that we want to compute in this threadblock
  const uint cCol = blockIdx.x;
  const uint cRow = blockIdx.y;

  // allocate buffer for current block in fast shared mem
  // shared mem is shared between all threads in a block
  __shared__ float As[BLOCK_SIZE * BLOCK_SIZE];
  __shared__ float Bs[BLOCK_SIZE * BLOCK_SIZE];

  // the inner row & col that we're accessing in this thread
  const uint threadCol = threadIdx.x % BLOCK_SIZE;
  const uint threadRow = threadIdx.x / BLOCK_SIZE;

  // advance pointers to the starting positions
  A += cRow * BLOCK_SIZE * K;                    // row=cRow, col=0
  B += cCol * BLOCK_SIZE;                        // row=0, col=cCol
  C += cRow * BLOCK_SIZE * N + cCol * BLOCK_SIZE; // row=cRow, col=cCol

  float tmp = 0.0;
  for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCK_SIZE) {
    // Have each thread load one of the elements in A & B
    // Make the threadCol (=threadIdx.x) the consecutive index
    // to allow global memory access coalescing
    As[threadRow * BLOCK_SIZE + threadCol] = A[threadRow * K + threadCol];
    Bs[threadRow * BLOCK_SIZE + threadCol] = B[threadRow * N + threadCol];

    // block threads in this block until cache is fully populated
    __syncthreads();
    A += BLOCK_SIZE;
    B += BLOCK_SIZE * N;

    // execute the dotproduct on the currently cached block
    for (int dotIdx = 0; dotIdx < BLOCK_SIZE; ++dotIdx) {
      tmp += As[threadRow * BLOCK_SIZE + dotIdx] *
             Bs[dotIdx * BLOCK_SIZE + threadCol];
    }
    // need to sync again at the end, to avoid faster threads
    // fetching the next block into the cache before slower threads are done
    __syncthreads();
  }
  C[threadRow * N + threadCol] = tmp;
}