#!/usr/bin/env python3
#
# Wei Alexander Xin - wax1
#
"""
q4.py: Plot execution times for Part-B Q1/Q2/Q3
Generates two charts:
  - q4_without_unified.jpg (CPU + Q2 scenarios)
  - q4_with_unified.jpg    (CPU + Q3 scenarios)
"""

import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Timing data from batch run on Insomnia (A6000, CUDA 12.9)
# ============================================================
K_values = [1, 5, 10, 50, 100]

# Q1: CPU baseline times (seconds)
cpu_times = [0.001155, 0.003917, 0.005884, 0.026447, 0.051670]

# Q2: GPU without unified memory (seconds)
q2_scenario1 = [0.066098, 0.309232, 0.617994, 3.071301, 6.196604]  # 1 block, 1 thread
q2_scenario2 = [
    0.001322,
    0.006546,
    0.013048,
    0.064532,
    0.129598,
]  # 1 block, 256 threads
q2_scenario3 = [
    0.000023,
    0.000094,
    0.000181,
    0.000874,
    0.001750,
]  # N/256 blocks, 256 thr

# Q3: GPU with unified memory (seconds)
q3_scenario1 = [0.065797, 0.308506, 0.616446, 3.080942, 6.163072]  # 1 block, 1 thread
q3_scenario2 = [
    0.001322,
    0.006501,
    0.012986,
    0.064783,
    0.123869,
]  # 1 block, 256 threads
q3_scenario3 = [
    0.000022,
    0.000093,
    0.000183,
    0.000893,
    0.001773,
]  # N/256 blocks, 256 thr

# ============================================================
# Chart 1: Without unified memory (CPU + Q2)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(K_values, cpu_times, "o-", label="CPU (Q1)", linewidth=2)
ax.plot(K_values, q2_scenario1, "s-", label="GPU: 1 block, 1 thread", linewidth=2)
ax.plot(K_values, q2_scenario2, "^-", label="GPU: 1 block, 256 threads", linewidth=2)
ax.plot(
    K_values, q2_scenario3, "D-", label="GPU: N/256 blocks, 256 threads", linewidth=2
)
ax.set_xlabel("K (millions of elements)", fontsize=12)
ax.set_ylabel("Execution Time (sec)", fontsize=12)
ax.set_title("Vector Addition: CPU vs GPU (without Unified Memory)", fontsize=14)
ax.set_xscale("log")
ax.set_yscale("log")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("q4_without_unified.jpg", dpi=150)
plt.close()

# ============================================================
# Chart 2: With unified memory (CPU + Q3)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(K_values, cpu_times, "o-", label="CPU (Q1)", linewidth=2)
ax.plot(K_values, q3_scenario1, "s-", label="GPU UM: 1 block, 1 thread", linewidth=2)
ax.plot(K_values, q3_scenario2, "^-", label="GPU UM: 1 block, 256 threads", linewidth=2)
ax.plot(
    K_values, q3_scenario3, "D-", label="GPU UM: N/256 blocks, 256 threads", linewidth=2
)
ax.set_xlabel("K (millions of elements)", fontsize=12)
ax.set_ylabel("Execution Time (sec)", fontsize=12)
ax.set_title("Vector Addition: CPU vs GPU (with Unified Memory)", fontsize=14)
ax.set_xscale("log")
ax.set_yscale("log")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("q4_with_unified.jpg", dpi=150)
plt.close()

print("Charts saved: q4_without_unified.jpg, q4_with_unified.jpg")
