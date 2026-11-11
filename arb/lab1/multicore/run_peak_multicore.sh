#!/bin/bash

# Set source and executable names
SRC="./compute_roof_mt.c"
EXE="./compute_roof_mt.o"

# Detect architecture and set appropriate flags
ARCH=$(uname -m)
 if [ "$ARCH" = "x86_64" ]; then
      # x86_64 architecture
      gcc -O3 -fopenmp -mfma -march=native -fno-tree-vectorize -fno-tree-slp-vectorize "$SRC" -o "$EXE" || exit 1
  else
      # ARM architecture
      gcc -O3 -mcpu=cortex-a76 -ffp-contract=fast -fopenmp -fno-tree-vectorize -fno-tree-slp-vectorize "$SRC" -o "$EXE" || exit 1

  fi

# Run the executable
echo "Running scalar roof benchmark..."
./$EXE 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
