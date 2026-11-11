#!/bin/bash
# Simple build + run script for kernels

SIZE=${1:-1000000}   # default: 1M elements
ITERS=${2:-200}      # default: 200 iterations for compute kernels


ARCH=$(uname -m)
  if [ "$ARCH" = "x86_64" ]; then
      # x86_64 architecture
      gcc -O3 -fopenmp -march=native -ffast-math -fno-tree-vectorize -ftree-slp-vectorize kernels_mt.c -o kernels_mt.o -lm || exit 1
  else
      # ARM architecture
      gcc -O3 -mcpu=cortex-a76 -ffp-contract=fast -fopenmp -ffast-math -fno-tree-vectorize -ftree-slp-vectorize kernels_mt.c -o kernels_mt.o -lm || exit 1

  fi


./kernels_mt.o fma $SIZE $ITERS
./kernels_mt.o dot $SIZE $ITERS
./kernels_mt.o poly $SIZE $ITERS
./kernels_mt.o triad $SIZE $ITERS
