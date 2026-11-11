#!/bin/bash
# Set source and executable names
SRC="./scalar_roof/scalar_roof.c"
EXE="./scalar_roof/scalar_roof.o"

# Detect architecture and set appropriate flags
ARCH=$(uname -m)
if [ "$ARCH" = "x86_64" ]; then
    # x86_64 architecture
    FLAGS="-O3 -mfma -fno-tree-vectorize -ftree-slp-vectorize -march=native"
else
    # ARM architecture
    FLAGS="-O3 -mcpu=cortex-a76 -ffp-contract=fast -fno-tree-vectorize -ftree-slp-vectorize"

fi

# Compile with optimizations and NO vectorization
gcc $FLAGS "$SRC" -o "$EXE" || exit 1

# Run the executable
echo "Running scalar roof benchmark..."
./$EXE 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
