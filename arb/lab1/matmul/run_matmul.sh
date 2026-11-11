#!/bin/bash
# Usage: ./run_matmul.sh 1 512

VERSION=$1
SIZE=${2:-512}

if [ -z "$VERSION" ] || [ -z "$SIZE" ]; then
    echo "Usage: $0 <version_number> [matrix_size]"
    exit 1
fi

SOURCE="matmul_${VERSION}.c"
if [ ! -f "$SOURCE" ]; then
    echo "Error: matrix multiplication version '$VERSION' does not exist ($SOURCE not found)." >&2
    exit 1
fi

ARCH=$(uname -m)
  if [ "$ARCH" = "x86_64" ]; then
      # x86_64 architecture
      gcc -O3 -march=native -fno-tree-vectorize -ftree-slp-vectorize "$SOURCE" main.c -o matmul_exec.o || exit 1
  else
      # ARM architecture
      gcc -O3 -mcpu=cortex-a76 -ffp-contract=fast -fopenmp -fno-tree-vectorize -ftree-slp-vectorize "$SOURCE" main.c -o matmul_exec.o || exit 1

  fi

./matmul_exec.o "$VERSION" "$SIZE"
