#!/bin/bash
# Download GSM8K test set for evaluation
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="${SCRIPT_DIR}/gsm8k"

mkdir -p "$TARGET_DIR"

echo "Downloading GSM8K test set..."
curl -L "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl" \
    -o "$TARGET_DIR/test.jsonl"

N_PROBLEMS=$(wc -l < "$TARGET_DIR/test.jsonl")
echo "Downloaded ${N_PROBLEMS} problems to ${TARGET_DIR}/test.jsonl"

# Quick sanity check
FIRST_Q=$(python3 -c "import json; print(json.loads(open('${TARGET_DIR}/test.jsonl').readline())['question'][:80])")
echo "First question: ${FIRST_Q}..."
echo "Done."
