/// Wei Alexander Xin - wax1
///
/// q1.cpp: CPU Baseline, add two K-million-element arrays
/// For COMS E6998 Spring 2026
///
/// Usage: ./q1 <K>
///   K = number of millions of elements (1, 5, 10, 50, 100)
///

#include <cstdio>
#include <cstdlib>
#include <sys/time.h>

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("Usage: %s <K>\n", argv[0]);
        printf("  K = number of millions of elements\n");
        return 1;
    }

    int K = atoi(argv[1]);
    long N = (long)K * 1000000;

    // Allocate arrays
    float* A = (float*)malloc(N * sizeof(float));
    float* B = (float*)malloc(N * sizeof(float));
    float* C = (float*)malloc(N * sizeof(float));

    if (!A || !B || !C) {
        printf("Memory allocation failed for N=%ld\n", N);
        return 1;
    }

    // Initialize
    for (long i = 0; i < N; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }

    // Time the addition: our humble CPU baseline... spoiler: GPU wins (eventually)
    double start = get_time();
    for (long i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
    double end = get_time();

    printf("K=%d, N=%ld, Time=%.6f sec\n", K, N, end - start);

    // Verify
    for (long i = 0; i < N; i++) {
        if (C[i] != 3.0f) {
            printf("Verification FAILED at index %ld\n", i);
            free(A); free(B); free(C);
            return 1;
        }
    }
    printf("Verification PASSED\n");

    free(A);
    free(B);
    free(C);
    return 0;
}
