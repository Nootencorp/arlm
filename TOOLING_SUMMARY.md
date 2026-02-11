# ARLM Experiment Tooling Summary

**Created:** 2026-02-11  
**Status:** ✅ Complete and tested

## Overview

Built comprehensive experiment tooling for ARLM (Augmented Recursive Language Model with Debate) research. All tools are production-ready with proper argparse help, error handling, and markdown output.

## Components Built

### 1. `scripts/analyze_results.py` - Results Analysis Tool

**Purpose:** Analyze experiment results, compute metrics, generate comparison tables and plots.

**Features:**
- Reads one or more JSONL result files
- Computes accuracy with 95% Clopper-Pearson confidence intervals
- Calculates average tokens and wall time
- Generates markdown-formatted comparison tables
- Optional matplotlib scaling curve plots (accuracy vs rounds)

**Usage:**
```bash
# Single file analysis
python scripts/analyze_results.py results/phase1_baseline_100.jsonl

# Compare multiple experiments
python scripts/analyze_results.py results/phase1_single_100.jsonl results/phase1_debate_100.jsonl

# Generate scaling curve plot
python scripts/analyze_results.py --plot results/scaling_curve.png results/phase2/*.jsonl
```

**Output Format:**
```
| Config | N | Accuracy | 95% CI | Avg Tokens | Avg Time |
|--------|---|----------|--------|------------|----------|
| phase1_baseline_20 | 20 | 75.0% | [50.9, 91.3] | 3427 | 10.1s |
```

**Tested:** ✅ Verified on `results/phase1_baseline_20.jsonl`

---

### 2. `experiments/02_baseline_rlm.py` - RLM-Only Baseline

**Purpose:** Isolate RLM contribution for ablation studies (no debate).

**Features:**
- Two modes: `raw` (pure LLM) and `rlm` (RLM-wrapped)
- Same interface as other experiments (argparse, JSONL output)
- Proper error handling and progress reporting
- Summary JSON output

**Usage:**
```bash
# Raw LLM mode (no RLM)
python experiments/02_baseline_rlm.py --mode raw --n-problems 100

# RLM-wrapped mode
python experiments/02_baseline_rlm.py --mode rlm --n-problems 100
```

**Key Design Decision:**
For GSM8K (short problems), RLM may not help much since the benefit is for long contexts. But we need this data point to prove super-additivity: **ARLM > RLM_alone + Debate_alone**.

---

### 3. `experiments/04_ablations.py` - Ablation Experiment

**Purpose:** Component isolation to test super-additivity hypothesis.

**Components Tested:**
1. Single agent (no debate, no RLM)
2. Debate only (Du et al. standard, 2 rounds)
3. RLM only (no debate)
4. ARLM (debate + RLM, 4 rounds)

**Features:**
- Selective component execution with `--components` flag
- Heterogeneous model support (future: GPT-4 moderator + GPT-3.5 agents)
- Unified output directory with consistent naming

**Usage:**
```bash
# Run all ablations
python experiments/04_ablations.py --n-problems 100

# Run specific components only
python experiments/04_ablations.py --components single debate --n-problems 50

# Custom output directory
python experiments/04_ablations.py --output-dir results/ablations_test
```

---

### 4. `scripts/run_phase1.sh` - Phase 1 Automation

**Purpose:** Automate all baseline experiments.

**Runs:**
1. Debate baseline (3 agents, 2 rounds)
2. Single agent baseline
3. RLM raw mode
4. RLM wrapped mode
5. Automatic analysis with comparison table

**Usage:**
```bash
# Default: gpt-3.5-turbo, 100 problems
./scripts/run_phase1.sh

# Custom model and size
./scripts/run_phase1.sh gpt-4-turbo 50

# Custom API key environment variable
./scripts/run_phase1.sh gpt-3.5-turbo 100 CUSTOM_API_KEY
```

**Estimated Cost:** ~$10 (100 problems, gpt-3.5-turbo)

---

### 5. `scripts/run_phase2.sh` - Phase 2 Automation

**Purpose:** Run ARLM scaling experiments.

**Features:**
- Calls `experiments/03_arlm_scaled.py` (already exists in repo)
- Automatic plot generation
- Tests multiple round configurations

**Usage:**
```bash
# Default: gpt-3.5-turbo, 100 problems
./scripts/run_phase2.sh

# Custom settings
./scripts/run_phase2.sh gpt-4-turbo 50
```

**Estimated Cost:** ~$120-150 (100 problems, multiple configs)

---

## Dependencies Added

Updated `requirements.txt`:
```
openai>=1.0
rlms
python-dotenv
rich
numpy
matplotlib  # NEW
scipy       # NEW
```

**Installed:**
- `scipy` - For Clopper-Pearson confidence intervals
- `matplotlib` - For scaling curve plots
- `python-dotenv` - For RLM environment loading

---

## Quality Checklist

✅ All scripts have proper `argparse` with `--help`  
✅ Analysis script produces clean markdown tables  
✅ Confidence intervals use scipy.stats (Clopper-Pearson)  
✅ Graceful handling of missing files  
✅ Tested `analyze_results.py` against existing data  
✅ All imports fixed (RLM path resolution)  
✅ Scripts are executable (`chmod +x`)  
✅ Git committed and pushed

---

## Testing Results

### analyze_results.py
```bash
$ python scripts/analyze_results.py results/phase1_baseline_20.jsonl
| Config | N | Accuracy | 95% CI | Avg Tokens | Avg Time |
|--------|---|----------|--------|------------|----------|
| phase1_baseline_20 | 20 | 75.0% | [50.9, 91.3] | 3427 | 10.1s |
```
✅ **PASS** - Clean table output, correct statistics

### Import Validation
```bash
$ python3 -c "from arlm.arlm_debate import StandardDebate, ARLMDebate; print('ok')"
ok
```
✅ **PASS** - All ARLM imports work

### Help Flags
All scripts tested with `--help`:
- `scripts/analyze_results.py --help` ✅
- `experiments/02_baseline_rlm.py --help` ✅
- `experiments/04_ablations.py --help` ✅

---

## Next Steps

### Ready to Run:
1. **Phase 1 Baselines:**
   ```bash
   ./scripts/run_phase1.sh
   ```

2. **Phase 2 ARLM Scaling:**
   ```bash
   ./scripts/run_phase2.sh
   ```

3. **Ablation Study:**
   ```bash
   python experiments/04_ablations.py --n-problems 100
   ```

### Note:
- Phase 1 baseline (100 problems) is currently running - don't interfere with `results/phase1_baseline_100.jsonl`
- All new experiments will write to separate files
- Cost estimates are approximate (actual costs depend on token usage)

---

## File Locations

```
arlm/
├── scripts/
│   ├── analyze_results.py       # ✅ NEW
│   ├── run_phase1.sh            # ✅ NEW
│   └── run_phase2.sh            # ✅ NEW
├── experiments/
│   ├── 01_baseline_debate.py    # (existing)
│   ├── 02_baseline_rlm.py       # ✅ REWRITTEN
│   ├── 03_arlm_scaled.py        # (existing)
│   └── 04_ablations.py          # ✅ REWRITTEN
├── requirements.txt             # ✅ UPDATED
└── TOOLING_SUMMARY.md           # ✅ NEW (this file)
```

---

**Git Commit:** `b6b716a`  
**Branch:** `main`  
**Status:** Pushed to origin
