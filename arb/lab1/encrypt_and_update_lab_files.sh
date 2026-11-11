#!/usr/bin/env bash

set -euo pipefail
umask 077

usage() {
    echo "Usage:"
    echo "  ./encrypt_submission.sh <group_number>"
    echo
    echo "Example:"
    echo "  ./encrypt_submission.sh 12"
    echo
    echo "This creates:"
    echo "  group_12_submission.enc"
}

if (( $# != 1 )); then
    echo "Error: exactly one group number must be provided."
    echo
    usage
    exit 1
fi

GROUP_NUMBER="$1"

# Accept only positive integer group numbers.
if [[ ! "$GROUP_NUMBER" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: the group number must be a positive integer."
    echo
    usage
    exit 1
fi

# Directory containing this script.
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# lab1_template must be beside the current lab directory.
PARENT_DIR="$(dirname "$DIR")"
TEMPLATE_DIR="$PARENT_DIR/lab1_template"

PUBLIC_KEY="$DIR/ta_public.pem"
OUTPUT_NAME="group_${GROUP_NUMBER}_submission.enc"
OUTPUT="$DIR/$OUTPUT_NAME"

TMPDIR_WORK="$(mktemp -d)"
ARCHIVE="$TMPDIR_WORK/submission.tar.gz"
AES_KEY="$TMPDIR_WORK/aes.key"
ENCRYPTED_ARCHIVE="$TMPDIR_WORK/archive.enc"
ENCRYPTED_KEY="$TMPDIR_WORK/key.enc"
PACKAGE="$TMPDIR_WORK/$OUTPUT_NAME"
TEMPLATE_COPY="$TMPDIR_WORK/lab1_template"

cleanup() {
    rm -rf -- "$TMPDIR_WORK"
}

trap cleanup EXIT

# Check the required commands.
for command_name in openssl tar find grep cp mkdir mv; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "Error: required command '$command_name' was not found."
        exit 1
    fi
done

# Check the public key.
if [[ ! -f "$PUBLIC_KEY" ]]; then
    echo "Error: ta_public.pem was not found."
    echo "Expected location:"
    echo "  $PUBLIC_KEY"
    exit 1
fi

if ! openssl pkey \
    -pubin \
    -in "$PUBLIC_KEY" \
    -noout >/dev/null 2>&1; then

    echo "Error: ta_public.pem is not a valid public key."
    exit 1
fi

# Check whether this group's encrypted output already exists.
if [[ -e "$OUTPUT" ]]; then
    echo "Error: the encrypted submission already exists:"
    echo "  $OUTPUT"
    echo
    echo "Move or remove the existing file before running the script again."
    exit 1
fi

# Check that lab1_template exists.
if [[ ! -d "$TEMPLATE_DIR" ]]; then
    echo "Error: the lab template directory was not found."
    echo "Expected location:"
    echo "  $TEMPLATE_DIR"
    exit 1
fi

# Check whether any file in lab1_template was modified
# during the previous 8 hours (480 minutes).
RECENT_FILE="$(
    find "$TEMPLATE_DIR" \
        -type f \
        -mmin -480 \
        -print \
        -quit
)"

if [[ -n "$RECENT_FILE" ]]; then
    echo "Error: the lab template cannot be used because a file"
    echo "inside it was modified during the last 8 hours:"
    echo "  $RECENT_FILE"
    echo
    echo "No files have been encrypted or deleted."
    exit 1
fi

# Prevent the template from overwriting protected files.
if [[ -e "$TEMPLATE_DIR/ta_public.pem" ]]; then
    echo "Error: lab1_template must not contain ta_public.pem."
    echo "Remove it from:"
    echo "  $TEMPLATE_DIR"
    exit 1
fi

TEMPLATE_ENCRYPTED_FILE="$(
    find "$TEMPLATE_DIR" \
        -maxdepth 1 \
        -type f \
        -name "group_*_submission.enc" \
        -print \
        -quit
)"

if [[ -n "$TEMPLATE_ENCRYPTED_FILE" ]]; then
    echo "Error: lab1_template contains an encrypted submission:"
    echo "  $TEMPLATE_ENCRYPTED_FILE"
    echo
    echo "Remove it before running this script."
    exit 1
fi

# Create a temporary copy of the validated template before
# performing encryption or deleting student files.
mkdir "$TEMPLATE_COPY"

if ! cp -a "$TEMPLATE_DIR"/. "$TEMPLATE_COPY"/; then
    echo "Error: the lab template could not be copied."
    echo "No files have been encrypted or deleted."
    exit 1
fi

echo "Group number: $GROUP_NUMBER"
echo "Output file: $OUTPUT_NAME"
echo
echo "Compressing files..."

# Compress everything in the script directory, including hidden files.
# Exclude the public key and all previous encrypted submissions.
tar -czf "$ARCHIVE" \
    --exclude="./ta_public.pem" \
    --exclude="./group_*_submission.enc" \
    -C "$DIR" .

# Verify the compressed archive.
if ! tar -tzf "$ARCHIVE" >/dev/null; then
    echo "Error: archive verification failed."
    exit 1
fi

echo "Encrypting files..."

# Generate a random 256-bit AES key.
openssl rand -out "$AES_KEY" 32

# Encrypt the archive with AES-256.
openssl enc \
    -aes-256-cbc \
    -pbkdf2 \
    -salt \
    -in "$ARCHIVE" \
    -out "$ENCRYPTED_ARCHIVE" \
    -pass "file:$AES_KEY"

# Encrypt the AES key with the TA public key.
openssl pkeyutl \
    -encrypt \
    -pubin \
    -inkey "$PUBLIC_KEY" \
    -pkeyopt rsa_padding_mode:oaep \
    -pkeyopt rsa_oaep_md:sha256 \
    -pkeyopt rsa_mgf1_md:sha256 \
    -in "$AES_KEY" \
    -out "$ENCRYPTED_KEY"

# Package the encrypted archive and encrypted AES key.
tar -cf "$PACKAGE" \
    -C "$TMPDIR_WORK" \
    archive.enc \
    key.enc

echo "Verifying encrypted output..."

if [[ ! -s "$PACKAGE" ]]; then
    echo "Error: encrypted output is empty."
    exit 1
fi

if ! tar -tf "$PACKAGE" | grep -qx "archive.enc"; then
    echo "Error: encrypted archive is missing from the package."
    exit 1
fi

if ! tar -tf "$PACKAGE" | grep -qx "key.enc"; then
    echo "Error: encrypted AES key is missing from the package."
    exit 1
fi

# Move the verified encrypted package into the lab directory.
mv -- "$PACKAGE" "$OUTPUT"

if [[ ! -s "$OUTPUT" ]]; then
    echo "Error: the final encrypted file was not created."
    exit 1
fi

echo "Encryption verified."
echo "Cleaning original files..."

# Remove everything except the public key and encrypted output.
find "$DIR" \
    -mindepth 1 \
    -maxdepth 1 \
    ! -name "ta_public.pem" \
    ! -name "$OUTPUT_NAME" \
    -exec rm -rf -- {} +

echo "Restoring the clean lab template..."

# Recursively copy all template contents, including hidden files.
if ! cp -a "$TEMPLATE_COPY"/. "$DIR"/; then
    echo "Error: the submission was encrypted successfully, but the"
    echo "clean lab template could not be restored."
    echo
    echo "The encrypted submission is safe at:"
    echo "  $OUTPUT"
    exit 1
fi

echo
echo "Done."
echo "Encrypted submission for group $GROUP_NUMBER:"
echo "  $OUTPUT"
echo
echo "A clean copy of lab1_template has been restored in:"
echo "  $DIR"