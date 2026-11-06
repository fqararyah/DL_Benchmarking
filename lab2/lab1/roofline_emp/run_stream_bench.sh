#!/bin/bash
# Script to build and run the STREAM benchmark, then report Triad bandwidth in GiB/s

# Path to STREAM source
STREAM_DIR="./stream"
STREAM_SRC="$STREAM_DIR/stream.c"
STREAM_EXE="$STREAM_DIR/stream.o"

# Build STREAM
echo "Building STREAM benchmark..."
ARCH=$(uname -m)
  if [ "$ARCH" = "x86_64" ]; then
      # x86_64 architecture
      gcc -O3 -fopenmp -DSTREAM_ARRAY_SIZE=100000000 -DNTIMES=10 "$STREAM_SRC" -o "$STREAM_EXE"
  else
      # ARM architecture
      gcc -O3 -mcpu=cortex-a76 -ffp-contract=fast -fopenmp -fno-tree-vectorize -ftree-slp-vectorize "$STREAM_SRC" -o "$STREAM_EXE"

  fi

# Run STREAM
echo "Running STREAM benchmark..."
OUTPUT=$("$STREAM_EXE")

# Print full output (optional)
echo "$OUTPUT"

# Extract Triad bandwidth (in MB/s) and convert to GiB/s
# Extract Triad bandwidth (in MB/s)
BW_MB=$(echo "$OUTPUT" | grep -m1 "Triad:" | awk '{print $2}')

# Convert to GB/s (decimal)
if [[ -n "$BW_MB" ]]; then
    BW_GB=$(awk -v mb="$BW_MB" 'BEGIN {printf "%.2f", mb / 1000}')
    echo ""
    echo "--------------------------------------------------"
    echo "Sustained memory bandwidth (Triad): ${BW_GB} GB/s"
    echo "--------------------------------------------------"
else
    echo "⚠️  Could not parse Triad bandwidth — check STREAM output format."
fi
