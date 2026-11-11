#!/bin/bash
# make_executable.sh - make all scripts and programs of Lab 2 executable.
# Use it if a command says "Permission denied".
# Usage: bash make_executable.sh      (works even if this file is not executable)

cd "$(dirname "$0")"          # the lab2 folder

find . -type f \( -name "*.sh" -o -name "*.py" \) -exec chmod +x {} +   # all scripts
chmod +x kernels/matmul/matmul_exec kernels/triad/triad                  # the two apps

echo "Done: all scripts and programs in $(pwd) are executable."
