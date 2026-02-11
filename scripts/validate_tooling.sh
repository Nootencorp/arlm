#!/bin/bash
# Validate ARLM experiment tooling

set -e

echo "=== ARLM Tooling Validation ==="
echo ""

# Check Python dependencies
echo "1. Checking Python dependencies..."
python3 -c "import scipy; import matplotlib; import numpy; print('  ✅ scipy, matplotlib, numpy')"
python3 -c "import openai; import rich; print('  ✅ openai, rich')"
python3 -c "from dotenv import load_dotenv; print('  ✅ python-dotenv')"

echo ""

# Check ARLM imports
echo "2. Checking ARLM imports..."
python3 -c "from arlm.arlm_debate import StandardDebate, ARLMDebate, extract_answer; print('  ✅ arlm.arlm_debate')"

echo ""

# Check scripts exist and are executable
echo "3. Checking scripts..."
for script in scripts/analyze_results.py scripts/run_phase1.sh scripts/run_phase2.sh; do
    if [ -x "$script" ]; then
        echo "  ✅ $script (executable)"
    else
        echo "  ❌ $script (not executable or missing)"
        exit 1
    fi
done

echo ""

# Check experiment scripts
echo "4. Checking experiment scripts..."
for exp in experiments/01_baseline_debate.py experiments/02_baseline_rlm.py experiments/03_arlm_scaled.py experiments/04_ablations.py; do
    if [ -f "$exp" ]; then
        echo "  ✅ $exp"
    else
        echo "  ❌ $exp (missing)"
        exit 1
    fi
done

echo ""

# Check GSM8K data
echo "5. Checking GSM8K benchmark data..."
if [ -f "benchmarks/gsm8k/test.jsonl" ]; then
    lines=$(wc -l < benchmarks/gsm8k/test.jsonl)
    echo "  ✅ benchmarks/gsm8k/test.jsonl ($lines problems)"
else
    echo "  ❌ benchmarks/gsm8k/test.jsonl (missing)"
    exit 1
fi

echo ""

# Test analyze_results.py
echo "6. Testing analyze_results.py..."
if [ -f "results/phase1_baseline_20.jsonl" ]; then
    python3 scripts/analyze_results.py results/phase1_baseline_20.jsonl > /dev/null 2>&1
    echo "  ✅ Analysis script works"
else
    echo "  ⚠️  No test data available (results/phase1_baseline_20.jsonl missing)"
fi

echo ""
echo "=== Validation Complete ==="
echo "All tooling is ready for experiments!"
echo ""
echo "Next steps:"
echo "  1. Run Phase 1 baselines: ./scripts/run_phase1.sh"
echo "  2. Run Phase 2 scaling: ./scripts/run_phase2.sh"
echo "  3. Run ablations: python experiments/04_ablations.py --n-problems 100"
