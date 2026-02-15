# wax1
import sys
import time
import numpy as np

"""
C5: The "Efficient" Python Dot Product!
Using np.dot leverages highly optimized BLAS libraries under the hood.
Restoring my faith in Python.
"""

def main():
  if len(sys.argv) != 3:
    # At least it's the same each time
    print(f"Usage: {sys.argv[0]} <vector_size> <repetitions>")
    return

  N = int(sys.argv[1])
  reps = int(sys.argv[2])

  # Numpy do your thing
  A = np.ones(N, dtype=np.float32)
  B = np.ones(N, dtype=np.float32)

  times = []

  for i in range(reps):
    start = time.perf_counter()
    
    # Two letters can do so much
    result = np.dot(A, B)
    
    end = time.perf_counter()
    times.append(end - start)

  # Standard "second half" mean computation
  start_index = reps // 2
  avg_time = sum(times[start_index:]) / (reps - start_index)

  # Memory and FLOP stats
  bandwidth = (2.0 * N * 4) / (avg_time * 1e9)
  throughput = (2.0 * N) / avg_time

  # Format a little
  print(f"N: {N} <T>: {avg_time:.6f} sec B: {bandwidth:.3f} GB/sec F: {throughput:.3e} FLOP/sec")

if __name__ == "__main__":
  main()