/// Wei Alexander Xin - wax1
///
/// q3.cu: GPU Vector Add WITH Unified Memory (cudaMallocManaged)
/// For COMS E6998 Spring 2026
///
/// Usage: ./q3 <K> <scenario>
///   K = number of millions of elements (1, 5, 10, 50, 100)
///   scenario = 1, 2, or 3
///     1: one block, 1 thread
///     2: one block, 256 threads
///     3: N/256 blocks, 256 threads per block
///

#include <cstdio>
#include <cstdlib>

__global__ void vecAdd(const float* A, const float* B, float* C, long N) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x * blockDim.x;
    for (long i = idx; i < N; i += stride) {
        C[i] = A[i] + B[i];
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        printf("Usage: %s <K> <scenario>\n", argv[0]);
        return 1;
    }

    int K = atoi(argv[1]);
    int scenario = atoi(argv[2]);
    long N = (long)K * 1000000;
    size_t size = N * sizeof(float);

    // Unified memory, one pointer to rule them all! runtime handles page migration
    float *A, *B, *C;
    cudaMallocManaged(&A, size);
    cudaMallocManaged(&B, size);
    cudaMallocManaged(&C, size);

    // Initialize on CPU -> data migrates to GPU on kernel launch
    for (long i = 0; i < N; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }

    // Configure launch parameters
    int threads, blocks;
    switch (scenario) {
        case 1: threads = 1;   blocks = 1; break;
        case 2: threads = 256; blocks = 1; break;
        case 3: threads = 256; blocks = (N + 255) / 256; break;
        default:
            printf("Invalid scenario: %d\n", scenario);
            return 1;
    }

    // Warm up
    vecAdd<<<blocks, threads>>>(A, B, C, N);
    cudaDeviceSynchronize();

    // Time the kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    vecAdd<<<blocks, threads>>>(A, B, C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    printf("K=%d, scenario=%d, blocks=%d, threads=%d, kernel_time=%.6f sec\n",
           K, scenario, blocks, threads, ms / 1000.0);

    // Verify (CPU reads from unified memory, will page-fault back from GPU)
    cudaDeviceSynchronize();
    int passed = 1;
    for (long i = 0; i < N; i++) {
        if (C[i] != 3.0f) {
            printf("Verification FAILED at index %ld\n", i);
            passed = 0;
            break;
        }
    }
    if (passed) printf("Verification PASSED\n");

    // Cleanup, use cudaFree, not free()
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
