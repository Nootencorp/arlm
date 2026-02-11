#!/bin/bash
# Run Phase 2: ARLM scaling experiments
# Estimated cost: ~$120-150

set -euo pipefail

MODEL="${1:-gpt-3.5-turbo}"
N="${2:-100}"
API_KEY_ENV="${3:-OPENAI_API_KEY}"

echo "=== Phase 2: ARLM Scaling ==="
echo "Model: $MODEL, N=$N problems"
echo "API Key from: $API_KEY_ENV"
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Check if 03_arlm_scaled.py exists
if [ ! -f "experiments/03_arlm_scaled.py" ]; then
    echo "ERROR: experiments/03_arlm_scaled.py not found!"
    echo "This script expects the ARLM scaling experiment to be implemented."
    echo "It should test multiple configurations:"
    echo "  - Standard debate (2, 4, 8 rounds)"
    echo "  - ARLM with summary strategy (2, 4, 8 rounds)"
    echo "  - ARLM with RLM strategy (2, 4, 8 rounds)"
    exit 1
fi

# Run the scaling experiment
python experiments/03_arlm_scaled.py \
    --model "$MODEL" --n-problems "$N" \
    --api-key-env "$API_KEY_ENV" \
    --output-dir "results/phase2"

echo ""
echo "========================================"

# Analysis
echo "--- Phase 2 Analysis ---"
python scripts/analyze_results.py --plot results/phase2/scaling_curve.png results/phase2/*.jsonl

echo ""
echo "=== Phase 2 Complete ==="
echo "Results saved in: results/phase2/"
echo "Scaling curve: results/phase2/scaling_curve.png"
