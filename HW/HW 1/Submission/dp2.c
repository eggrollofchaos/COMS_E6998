// wax1
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/**
 * The "Faster" Unrolled Dot Product.
 * We are manually unrolling the loop by a factor of 4.
 * Why? Because we like to help the CPU's pipeline feel less lonely.
 */
float dpunroll(long N, float *pA, float *pB) {
  float R = 0.0;
  long j;

  // We assume N is a multiple of 4 as per the prompt's implied logic
  for (j = 0; j < N; j += 4) {
    R += pA[j] * pB[j] + pA[j + 1] * pB[j + 1] + 
         pA[j + 2] * pB[j + 2] + pA[j + 3] * pB[j + 3];
  }

  return R;
}

int main(int argc, char *argv[]) {
  
  // Checking arguments just because
  if (argc != 3) {
    printf("Usage: %s <vector_size> <repetitions>\n", argv[0]);
    return 1;
  }

  long N = atol(argv[1]);
  int reps = atoi(argv[2]);

  // Requesting lots of memory
  float *pA = (float *)malloc(N * sizeof(float));
  float *pB = (float *)malloc(N * sizeof(float));

  if (pA == NULL || pB == NULL) {
    fprintf(stderr, "The OS said 'No' to your memory request. Try a bigger VM!\n");
    return 1;
  }

  // Filling these vectors with ones, forever
  for (long i = 0; i < N; i++) {
    pA[i] = 1.0f;
    pB[i] = 1.0f;
  }

    double *times = (double *)malloc(reps * sizeof(double));

  // Perform measurements.........
  for (int i = 0; i < reps; i++) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    // Prevent Dead Code Elimination using 'volatile' per assignment
    volatile float result = dpunroll(N, pA, pB);
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    times[i] = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
  }

  // We ignore the first half of the reps to give time for CPU to warm up
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

  // Printing huge numbers again
  printf("N: %ld <T>: %f sec B: %f GB/sec F: %f FLOP/sec\n", N, avg_time, bandwidth, throughput);

  // Clean up
  free(pA);
  free(pB);
  free(times);

  return 0;
}