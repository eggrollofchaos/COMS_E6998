/// Wei Alexander Xin - wax1
///
/// q2.cu: GPU Vector Add WITHOUT Unified Memory (cudaMalloc + cudaMemcpy)
/// For COMS E6998 Spring 2026
///
/// Usage: ./q2 <K> <scenario>
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

    // Allocate host memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    // Initialize
    for (long i = 0; i < N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy host -> device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 1 thread = expensive for loop
    // 256 threads = one SM
    // full grid = actually using the GPU
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
    vecAdd<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    // Time the kernel, cool
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    vecAdd<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    printf("K=%d, scenario=%d, blocks=%d, threads=%d, kernel_time=%.6f sec\n",
           K, scenario, blocks, threads, ms / 1000.0);

    // Copy result back
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verify
    int passed = 1;
    for (long i = 0; i < N; i++) {
        if (h_C[i] != 3.0f) {
            printf("Verification FAILED at index %ld\n", i);
            passed = 0;
            break;
        }
    }
    if (passed) printf("Verification PASSED\n");

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
