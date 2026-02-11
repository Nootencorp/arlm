#!/bin/bash
# Run all Phase 1 experiments (baselines)
# Estimated cost: ~$10

set -euo pipefail

MODEL="${1:-gpt-3.5-turbo}"
N="${2:-100}"
API_KEY_ENV="${3:-OPENAI_API_KEY}"

echo "=== Phase 1: Baseline Experiments ==="
echo "Model: $MODEL, N=$N problems"
echo "API Key from: $API_KEY_ENV"
echo ""

# Get script directory (where this script is located)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# 1A: Debate baseline (3 agents, 2 rounds)
echo "--- Experiment 1A: Debate baseline ---"
python experiments/01_baseline_debate.py \
    --model "$MODEL" --n-agents 3 --n-rounds 2 --n-problems "$N" \
    --api-key-env "$API_KEY_ENV" \
    --output "results/phase1_debate_${N}.jsonl"

echo ""

# 1B: Single agent baseline
echo "--- Experiment 1B: Single agent ---"  
python experiments/01_baseline_debate.py \
    --model "$MODEL" --n-agents 1 --n-rounds 1 --n-problems "$N" \
    --api-key-env "$API_KEY_ENV" \
    --output "results/phase1_single_${N}.jsonl"

echo ""

# 1C: RLM-only baseline (raw mode)
echo "--- Experiment 1C: RLM raw mode ---"
python experiments/02_baseline_rlm.py \
    --model "$MODEL" --mode raw --n-problems "$N" \
    --api-key-env "$API_KEY_ENV" \
    --output "results/phase1_raw_${N}.jsonl"

echo ""

# 1D: RLM-only baseline (rlm mode)
echo "--- Experiment 1D: RLM mode ---"
python experiments/02_baseline_rlm.py \
    --model "$MODEL" --mode rlm --n-problems "$N" \
    --api-key-env "$API_KEY_ENV" \
    --output "results/phase1_rlm_${N}.jsonl"

echo ""
echo "========================================"

# Analysis
echo "--- Phase 1 Analysis ---"
python scripts/analyze_results.py \
    "results/phase1_single_${N}.jsonl" \
    "results/phase1_debate_${N}.jsonl" \
    "results/phase1_raw_${N}.jsonl" \
    "results/phase1_rlm_${N}.jsonl"

echo ""
echo "=== Phase 1 Complete ==="
echo "Results saved in: results/phase1_*_${N}.jsonl"
