// wax1
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mkl_cblas.h>

/**
 * C3: The "Professional" Dot Product using Intel MKL.
 * We are using cblas_sdot because Intel engineers spent a lottt of time
 * making it faster than anything we could write in a weekend.
 */
float bdp(long N, float *pA, float *pB) {
  // It's the way to go for peak performance
  return cblas_sdot(N, pA, 1, pB, 1);
}

int main(int argc, char *argv[]) {
  
  // Someone is definitely going to need this helpful usage tip someday
  if (argc != 3) {
    printf("Usage: %s <vector_size> <repetitions>\n", argv[0]);
    return 1;
  }

  long N = atol(argv[1]);
  int reps = atoi(argv[2]);

  // Requesting memory for the heavy, heavy lifting
  float *pA = (float *)malloc(N * sizeof(float));
  float *pB = (float *)malloc(N * sizeof(float));

  if (pA == NULL || pB == NULL) {
    fprintf(stderr, "MKL needs memory to breathe. Get a bigger machine!\n");
    return 1;
  }

  // Hazing ritual
  for (long i = 0; i < N; i++) {
    pA[i] = 1.0f;
    pB[i] = 1.0f;
  }

  double *times = (double *)malloc(reps * sizeof(double));

  // Perform measurements........................
  for (int i = 0; i < reps; i++) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    // Prevent Dead Code Elimination using 'volatile' per assignment
    volatile float result = bdp(N, pA, pB);
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    times[i] = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
  }

  // Averaging the second half because first impressions (warm-up) are misleading
  double total_time = 0;
  int start_index = reps / 2;
  int count = reps - start_index;

  for (int i = start_index; i < reps; i++) {
    total_time += times[i];
  }
  double avg_time = total_time / count;

  // Throughput calculation:
  // Each index involves 1 multiplication and 1 addition = 2 FLOPs per element
  // Total FLOPs = 2 * N
  double bandwidth = (2.0 * N * sizeof(float)) / (avg_time * 1e9);
  double throughput = (2.0 * N) / avg_time;

  // Printing hugely
  printf("N: %ld <T>: %f sec B: %f GB/sec F: %f FLOP/sec\n", N, avg_time, bandwidth, throughput);

  // Clean up
  free(pA);
  free(pB);
  free(times);

  return 0;
}