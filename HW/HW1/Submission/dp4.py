# wax1
import sys
import time
import numpy as np

"""
C4: The "Patience Tester" Dot Product.
Implementing a dot product with a Python for-loop is a...
really great way to appreciate how fast C actually is.
"""

def dp(N, A, B):
  R = 0.0
  # This loop will take a while for N=300,000,000.
  # Grab a coffee, or really a whole meal.
  for j in range(0, N):
    R += A[j] * B[j]
  return R

def main():
  if len(sys.argv) != 3:
    # Writing this is definitely a compulsion
    print(f"Usage: {sys.argv[0]} <vector_size> <repetitions>")
    return

  N = int(sys.argv[1])
  reps = int(sys.argv[2])

  # Initializing NumPy arrays, gotta say I do miss Numpy sometimes
  A = np.ones(N, dtype=np.float32)
  B = np.ones(N, dtype=np.float32)

  times = []

  for i in range(reps):
    start = time.perf_counter()
    
    # In Python, we don't have 'volatile', but the interpreter 
    # won't optimize this loop away like a C compiler would.
    result = dp(N, A, B)
    
    end = time.perf_counter()
    times.append(end - start)

  # Use the second half to ignore "warm-up" (though Python stays pretty cool)
  start_index = reps // 2
  avg_time = sum(times[start_index:]) / (reps - start_index)

  # 2 arrays * N elements * 4 bytes
  bandwidth = (2.0 * N * 4) / (avg_time * 1e9)
  # 2 FLOPs per element
  throughput = (2.0 * N) / avg_time

  # Format a little
  print(f"N: {N} <T>: {avg_time:.6f} sec B: {bandwidth:.3f} GB/sec F: {throughput:.3e} FLOP/sec")

if __name__ == "__main__":
  main()