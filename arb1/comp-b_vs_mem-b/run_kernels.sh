#!/bin/bash
# Simple build + run script for kernels

SIZE=${1:-1000000}   # default: 1M elements
ITERS=${2:-200}      # default: 200 iterations for compute kernels


ARCH=$(uname -m)
  if [ "$ARCH" = "x86_64" ]; then
      # x86_64 architecture
      gcc -O3 -march=native -ffast-math -fno-tree-vectorize -ftree-slp-vectorize kernels.c -o kernels.o -lm || exit 1
  else
      # ARM architecture
      gcc -O3 -mcpu=cortex-a76 -ffp-contract=fast -fopenmp -fno-tree-vectorize -ftree-slp-vectorize kernels.c -o kernels.o -lm || exit 1

  fi

./kernels.o fma $SIZE $ITERS
./kernels.o dot $SIZE $ITERS
./kernels.o poly $SIZE $ITERS
./kernels.o triad $SIZE $ITERS
