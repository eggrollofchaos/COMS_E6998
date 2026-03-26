// wax1
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/**
 * Baseline dot product mini-benchmark.
 * Calculates the sum of pA[j] * pB[j] for j = 0 to N-1.
 * Very basic.
 */
float dp(long N, float *pA, float *pB) {
  float R = 0.0;
  int j;
  for (j = 0; j < N; j++) {
    R += pA[j] * pB[j];
  }
  return R;
}

int main(int argc, char *argv[]) {
  
  // As if anyone needs this
  if (argc != 3) {
    printf("Usage: %s <vector_size> <repetitions>\n", argv[0]);
    return 1;
  }

  long N = atol(argv[1]);
  int reps = atoi(argv[2]);

  // Allocate alllll the memory for vectors
  float *pA = (float *)malloc(N * sizeof(float));
  float *pB = (float *)malloc(N * sizeof(float));

  if (pA == NULL || pB == NULL) {
    fprintf(stderr, "Memory allocation failed!\n");
    return 1;
  }

  // Initialize vectors, imagine how tedious this would be if done manually
  for (long i = 0; i < N; i++) {
    pA[i] = 1.0f;
    pB[i] = 1.0f;
  }

  double *times = (double *)malloc(reps * sizeof(double));

  // Perform measurements...
  for (int i = 0; i < reps; i++) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    // Prevent Dead Code Elimination using 'volatile' per assignment
    volatile float result = dp(N, pA, pB);
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    times[i] = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
  }

  // Compute mean of the second half of repetitions
  double total_time = 0;
  int start_index = reps / 2;
  int count = reps - start_index;

  for (int i = start_index; i < reps; i++) {
    total_time += times[i];
  }
  double avg_time = total_time / count;

  // Bandwidth calculation:
  // Each dot product accesses 2 * N floats (pA and pB)
  // Each float is 4 bytes, total bytes = 2 * N * 4
  double bandwidth = (2.0 * N * sizeof(float)) / (avg_time * 1e9);

  // Throughput calculation:
  // Each index involves 1 multiplication and 1 addition = 2 FLOPs per element
  // Total FLOPs = 2 * N
  double throughput = (2.0 * N) / avg_time;

  // I wish the assignment would have allowed us to write it as GFLOPS,
  // Hard to read these numbers when written as FLOPS!
  printf("N: %ld <T>: %f sec B: %f GB/sec F: %f FLOP/sec\n", N, avg_time, bandwidth, throughput);

  // Print actual result
  float dot_product = dp(N, pA, pB);
  printf("Final Result: %f (Expected: %ld.0)\n", dot_product, N);

  // Clean up
  free(pA);
  free(pB);
  free(times);

  return 0;
}