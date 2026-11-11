#!/usr/bin/env bash

set -euo pipefail

# Directory containing this script.
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# lab1_template is beside the current lab directory.
TEMPLATE_DIR="$(dirname "$DIR")/lab1_template"

if (( $# != 0 )); then
    echo "Error: run this script without arguments:"
    echo "  ./update_from_template.sh"
    exit 1
fi

if [[ ! -d "$TEMPLATE_DIR" ]]; then
    echo "Error: lab1_template was not found."
    echo "Expected location:"
    echo "  $TEMPLATE_DIR"
    exit 1
fi

# Reject the template if any file was modified in the last 8 hours.
RECENT_FILE="$(
    find "$TEMPLATE_DIR" \
        -type f \
        -mmin -480 \
        -print \
        -quit
)"

if [[ -n "$RECENT_FILE" ]]; then
    echo "Error: the template cannot be used because a file"
    echo "was modified during the last 8 hours:"
    echo "  $RECENT_FILE"
    echo
    echo "No files were updated."
    exit 1
fi

echo "Updating files from:"
echo "  $TEMPLATE_DIR"

# Copy files, hidden files, and subdirectories recursively.
cp -a "$TEMPLATE_DIR"/. "$DIR"/

echo "Update completed successfully."