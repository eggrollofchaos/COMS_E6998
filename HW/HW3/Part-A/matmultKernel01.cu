/// Wei Alexander Xin - wax1
///
/// matmultKernel01.cu
/// For COMS E6998 Spring 2023
/// Instructor: Parajit Dube and Kaoutar El Maghraoui
/// Based on code from the CUDA Programming Guide
/// Modified by Wim Bohm and David Newman
/// Created: 2011-01-27
/// Last Modified: 2011-02-23 DVN
///
/// Multiplies two matrices using CUDA: A x B = C
///
/// Optimized kernel: each thread computes FOUR output values.
/// FOOTPRINT_SIZE = 32 (defined via -D flag), BLOCK_SIZE = 16.
/// Thread block is 16x16, but each block covers a 32x32 output tile.
///
/// Thread (ty, tx) computes:
///   C[row,       col      ]
///   C[row,       col + 16 ]
///   C[row + 16,  col      ]
///   C[row + 16,  col + 16 ]
///
/// Shared memory tiles are 16x16 (BLOCK_SIZE), loaded in two phases
/// per iteration to cover the 32-wide footprint.
///

#include "matmultKernel.h"

// Define a gpu kernel to perform matrix multiplication of A x B = C.
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C){

  // matrix blocks
  float *Asub, *Bsub, *Csub;
  // Putting these into registers speeds access.
  int thread_row = threadIdx.y;
  int thread_col = threadIdx.x;
  int block_row = blockIdx.y;
  int block_col = blockIdx.x;

  // Each THREAD BLOCK computes one sub matrix Csub of C
  // EACH THREAD creates its own matrix descriptor Csub

  // Note: FOOTPRINT_SIZE becomes 32 from Makefile
  Csub = &C.elements[C.stride * FOOTPRINT_SIZE * block_row + FOOTPRINT_SIZE * block_col];

  // Each thread claims a 2x2 patch of the output -- 4x the work, way less idle time
  float Cvalue00 = 0;  // (row,       col      )
  float Cvalue01 = 0;  // (row,       col + 16 )
  float Cvalue10 = 0;  // (row + 16,  col      )
  float Cvalue11 = 0;  // (row + 16,  col + 16 )

  // Loop over all sub matrices in block_row of A and block_col of B
  // required to compute Csub. Block multiply each pair of sub matrices
  // and accumulate results
  for (int m = 0;  m < (A.width / BLOCK_SIZE); ++m){
    // Get Asub and Bsub descriptors

    // Note: A row uses FOOTPRINT_SIZE stride (32 rows per block)
    Asub = &A.elements[A.stride * FOOTPRINT_SIZE * block_row + BLOCK_SIZE * m];
    Bsub = &B.elements[B.stride * BLOCK_SIZE * m + FOOTPRINT_SIZE * block_col];

    // Notice: every thread declares shared_A and shared_B in shared memory
    //         even though a thread block has only one shared_A and one shared_B
    __shared__ float shared_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float shared_B[BLOCK_SIZE][BLOCK_SIZE];

    // Doing 4 load-compute phases per iteration! covering 32x32 with 16x16 tiles

    // Phase 1
    // Load top 16x16 of A, left 16x16 of B
    shared_A[thread_row][thread_col] = Asub[thread_row * A.stride + thread_col];
    shared_B[thread_row][thread_col] = Bsub[thread_row * B.stride + thread_col];

    // Synchronize to ensure all elements are read
    __syncthreads();

    // Accumulate Cvalue00 (row, col)
#pragma unroll
    for(int e=0; e<BLOCK_SIZE; ++e)
       Cvalue00 += shared_A[thread_row][e] * shared_B[e][thread_col];

    // Synchronize before loading next tile
    __syncthreads();

    // Phase 2
    // Load right 16x16 of B (cols 16..31)
    shared_B[thread_row][thread_col] = Bsub[thread_row * B.stride + thread_col + BLOCK_SIZE];
    __syncthreads();

    // Accumulate Cvalue01 (row, col+16) using same A tile, shifted B
#pragma unroll
    for(int e=0; e<BLOCK_SIZE; ++e)
       Cvalue01 += shared_A[thread_row][e] * shared_B[e][thread_col];

    __syncthreads();

    // Phase 3
    // Load bottom 16x16 of A (rows 16..31)
    shared_A[thread_row][thread_col] = Asub[(thread_row + BLOCK_SIZE) * A.stride + thread_col];
    __syncthreads();

    // Accumulate Cvalue11 (row+16, col+16) using bottom A, right B
#pragma unroll
    for(int e=0; e<BLOCK_SIZE; ++e)
       Cvalue11 += shared_A[thread_row][e] * shared_B[e][thread_col];

    __syncthreads();

    // Phase 4
    // Reload left 16x16 of B (cols 0..15)
    shared_B[thread_row][thread_col] = Bsub[thread_row * B.stride + thread_col];
    __syncthreads();

    // Accumulate Cvalue10 (row+16, col) using bottom A, left B
#pragma unroll
    for(int e=0; e<BLOCK_SIZE; ++e)
       Cvalue10 += shared_A[thread_row][e] * shared_B[e][thread_col];

    // Synchronize to ensure all Cvalues have been incremented
    // before reading in the next shared_A AND shared_B BLOCKS
    __syncthreads();
  }

  // Write Csub to GLOBAL memory.
  // Each thread writes its own four cell values.
  Csub[thread_row * C.stride + thread_col]                             = Cvalue00;
  Csub[thread_row * C.stride + thread_col + BLOCK_SIZE]                = Cvalue01;
  Csub[(thread_row + BLOCK_SIZE) * C.stride + thread_col]              = Cvalue10;
  Csub[(thread_row + BLOCK_SIZE) * C.stride + thread_col + BLOCK_SIZE] = Cvalue11;
}
