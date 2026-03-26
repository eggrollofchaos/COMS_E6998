#!/bin/bash
#
# Wei Alexander Xin - wax1
#
#SBATCH --job-name=hw3
#SBATCH --account=edu
#SBATCH --partition=burst
#SBATCH --qos=burst
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=hw3-%j.out

echo "=== System Info ==="
echo "Hostname: $(hostname)"
echo "Date: $(date)"
lscpu | grep -E "Model name|Socket|Core|Thread"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
nvcc --version | tail -1
echo ""

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$HOME/.local/lib/python3.9/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

echo "=== Part A ==="
cd Part-A
make clean && make all 2>&1

echo ""
echo "--- Q1: vecadd00 (non-coalesced) ---"
for vpt in 500 1000 2000; do
    echo "ValuesPerThread=$vpt:"
    ./vecadd00 $vpt
done

echo ""
echo "--- Q2: vecadd01 (coalesced) ---"
for vpt in 500 1000 2000; do
    echo "ValuesPerThread=$vpt:"
    ./vecadd01 $vpt
done

echo ""
echo "--- Q3: matmult00 (baseline) ---"
for sz in 16 32 64; do
    echo "NumBlocks=$sz (matrix $((sz*16))x$((sz*16))):"
    ./matmult00 $sz
done

echo ""
echo "--- Q4: matmult01 (optimized, FOOTPRINT=32) ---"
for sz in 8 16 32; do
    echo "NumBlocks=$sz (matrix $((sz*32))x$((sz*32))):"
    ./matmult01 $sz
done

echo ""
echo "=== Part B ==="
cd ../Part-B
make clean && make all 2>&1

echo ""
for K in 1 5 10 50 100; do
    echo "--- K=$K million ---"
    echo "Q1 (CPU):"
    ./q1 $K
    for s in 1 2 3; do
        echo "Q2 scenario $s:"
        ./q2 $K $s
        echo "Q3 scenario $s:"
        ./q3 $K $s
    done
    echo ""
done

echo ""
echo "=== Part C ==="
cd ../Part-C
make clean && make all 2>&1
echo ""
echo "C1 (basic):"
./c1
echo "C2 (tiled):"
./c2
echo "C3 (cuDNN):"
./c3

echo ""
echo "=== Done ==="
