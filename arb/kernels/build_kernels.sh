#!/bin/bash
# build_kernels.sh — compile the two benchmark kernels used in Lab 2.
#   MatMul (compute-heavy app): kernels/matmul/matmul_exec   (N and RUNS are set in matmul/main.c)
#   Triad  (memory-heavy app) : kernels/triad/triad          (N and RUNS are set in triad/kernels.c)
# Usage: ./kernels/build_kernels.sh

set -e
cd "$(dirname "$0")"

echo "Building MatMul..."
gcc -O3 -fno-tree-vectorize -ftree-slp-vectorize matmul/matmul_5.c matmul/main.c -o matmul/matmul_exec

echo "Building Triad..."
gcc -O3 -march=native -ffast-math -fno-tree-vectorize -ftree-slp-vectorize triad/kernels.c -o triad/triad -lm

echo "Done."
