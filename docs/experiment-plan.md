# ARLM Experiment Plan v1

**Status:** Draft — awaiting Jeremy's review
**Date:** 2026-02-11
**Goal:** Demonstrate that combining RLM + adversarial debate produces compound gains beyond the sum of their parts

---

## Core Thesis

Du et al. (ICML 2024) showed multi-agent debate improves LLM reasoning but **capped at 3 agents × 2 rounds due to computational cost** (their words). Context grows as O(agents × rounds × response_length) — each round, every agent reads all other agents' prior responses.

RLM (Zhang et al. 2025) eliminates context limits via programmatic recursion through a REPL environment.

**ARLM hypothesis:** Remove the context ceiling from debate → accuracy keeps climbing past Du et al.'s 2-round limit → compound gains beyond either technique alone.

---

## Repo Structure

```
adversarial-rlm/
├── README.md                    # Paper / concept overview
├── LICENSE                      # MIT
├── requirements.txt
├── rlm/                         # Forked from alexzhang13/rlm-minimal
│   └── (minimal RLM implementation)
├── debate/                      # Forked from composable-models/llm_multiagent_debate
│   └── (Du et al. debate framework)
├── arlm/                        # OUR CONTRIBUTION — the integration layer
│   ├── arlm_debate.py           # RLM-wrapped debate: agents use REPL for context
│   ├── config.py                # Experiment configurations
│   └── utils.py
├── experiments/
│   ├── 01_baseline_debate.py    # Reproduce Du et al. exactly
│   ├── 02_baseline_rlm.py       # Reproduce RLM on same benchmarks
│   ├── 03_arlm_scaled.py        # ARLM at 2, 4, 6, 8 rounds
│   ├── 04_ablations.py          # Component ablations
│   └── configs/                 # YAML configs per experiment
├── benchmarks/
│   ├── gsm8k/                   # Grade School Math (1,319 test problems)
│   ├── mmlu/                    # Massive Multitask Language Understanding
│   └── arithmetic/              # Procedurally generated (from Du et al.)
├── results/                     # Raw results + analysis notebooks
│   └── analysis.ipynb
└── scripts/
    ├── run_all.sh               # Full experiment suite
    └── cost_estimate.py         # Pre-run cost calculator
```

---

## Experiment Design

### Phase 0: Setup (Day 1)
- Fork `alexzhang13/rlm-minimal` and `composable-models/llm_multiagent_debate`
- Download GSM8K test set (1,319 problems) and MMLU test set
- Verify both codebases run independently
- Set up results logging (JSON per run: config, accuracy, token count, cost, wall time)

### Phase 1: Reproduce Baselines (Days 2-3)

**Experiment 1A — Du et al. Debate Baseline**
- Config: 3 agents, 2 rounds, GSM8K full test set (1,319 problems)
- Model: `gpt-3.5-turbo` (what they used; still available, $0.50/1M in, $1.50/1M out)
- Expected result: ~89.0% accuracy (their reported number)
- Purpose: Prove our fork works and matches published results

**Experiment 1B — Du et al. on Arithmetic**
- Config: 3 agents, 2 rounds, 100 arithmetic problems (6 random integers, as in paper)
- Model: `gpt-3.5-turbo`
- Expected: ~72% (their reported, vs 40% single agent)

**Experiment 1C — Single-agent Baseline**
- Config: 1 agent, no debate, GSM8K
- Model: `gpt-3.5-turbo`
- Expected: ~78% (their reported single-agent number)

**Experiment 1D — RLM Baseline on GSM8K**
- Config: RLM wrapping single GPT-3.5-turbo, no debate
- Purpose: Measure RLM's contribution in isolation on the same benchmark
- Note: RLM was not tested on GSM8K in Zhang et al. — this is new data

**Success gate:** Experiments 1A-1C must match Du et al.'s reported numbers within ±2%.  If they don't, debug before proceeding.

### Phase 2: ARLM — Scale Debate Rounds (Days 4-6)

This is the core experiment. We use RLM to wrap the debate loop, storing all agent responses in the REPL environment as variables. Each agent uses RLM-style programmatic access to read prior rounds instead of cramming everything into context.

**Experiment 2A — ARLM Scaling Curve (GSM8K)**

| Config | Agents | Rounds | Context Strategy |
|--------|--------|--------|-----------------|
| debate-2r | 3 | 2 | Standard (Du et al. baseline) |
| debate-4r | 3 | 4 | Standard (will hit context limits) |
| arlm-2r | 3 | 2 | RLM-wrapped |
| arlm-4r | 3 | 4 | RLM-wrapped |
| arlm-6r | 3 | 6 | RLM-wrapped |
| arlm-8r | 3 | 8 | RLM-wrapped |

- Model: `gpt-3.5-turbo` (for comparability with Du et al.)
- Benchmark: GSM8K full test set
- Key metric: accuracy vs. rounds, with and without RLM
- **Expected result:** Standard debate degrades or fails at 4+ rounds (context overflow). ARLM continues improving. The delta is the paper.

**Experiment 2B — ARLM Scaling Curve (Arithmetic)**
- Same configs as 2A but on arithmetic (100 problems)
- Faster to run, good for iterating on the integration

**Experiment 2C — Agent Scaling**

| Config | Agents | Rounds | Context Strategy |
|--------|--------|--------|-----------------|
| arlm-3a-4r | 3 | 4 | RLM-wrapped |
| arlm-5a-4r | 5 | 4 | RLM-wrapped |
| arlm-7a-4r | 7 | 4 | RLM-wrapped |

- Purpose: Du et al. only tested 3 and 6 agents. With RLM handling context, we can scale agents too.

### Phase 3: Ablations (Days 7-8)

**Experiment 3A — Component Isolation**
- RLM only (no debate) on GSM8K → isolates RLM's contribution
- Debate only (no RLM) at max feasible rounds → isolates debate's contribution
- ARLM at same rounds → measures compound effect
- **Key test:** Is ARLM > RLM + Debate individually? (super-additivity)

**Experiment 3B — Heterogeneous Models**
- ARLM with mixed models in debate (e.g., GPT-3.5 + GPT-4o-mini)
- Ties to our paper's heterogeneous model hypothesis

**Experiment 3C — REPL State Utilization**
- Compare: agents just reading text summaries vs. agents with programmatic grep/slice access to REPL state
- Tests whether the REPL integration actually matters vs. just having more rounds

### Phase 4: Modern Models (Day 9-10, optional)

- Repeat Experiment 2A with `gpt-4o-mini` ($0.15/1M in, $0.60/1M out)
- Shows results generalize beyond GPT-3.5-turbo
- Potentially stronger absolute numbers

---

## Cost Estimate

### Token budget per GSM8K problem (debate, 3 agents, 2 rounds):
- System prompt: ~200 tokens
- Problem: ~100 tokens
- Each agent response: ~300 tokens
- Each round, each agent reads: problem + all prior responses
- Round 1: 3 agents × (200 + 100 + 0) input + 300 output = 1,800 in + 900 out
- Round 2: 3 agents × (200 + 100 + 900) input + 300 output = 3,600 in + 900 out
- **Total per problem (2 rounds): ~5,400 input + 1,800 output tokens**

### Phase 1 (baselines):
- 1A: 1,319 problems × 7,200 tokens avg = ~9.5M tokens → **~$7**
- 1B: 100 problems → **~$0.50**
- 1C: 1,319 problems × 600 tokens = ~0.8M tokens → **~$0.50**
- 1D: ~1,319 problems × ~1,500 tokens (RLM overhead) → **~$2**
- **Phase 1 total: ~$10**

### Phase 2 (ARLM scaling):
- Each additional round adds ~1,800 tokens per problem per round
- 6 configs × 1,319 problems × avg ~15,000 tokens = ~119M tokens
- With RLM overhead (recursive calls ~1.5x multiplier) = ~178M tokens
- **Phase 2 total: ~$120-150** (largest cost, the core experiment)

### Phase 3 (ablations):
- Subset of GSM8K (300 problems) × 5 configs → **~$20-30**

### Phase 4 (modern models):
- Same as Phase 2 but with gpt-4o-mini (4x cheaper) → **~$30-40**

### **Total estimated cost: $160-230**

### Cost optimization:
- Run Phase 2 on arithmetic (100 problems) first as a cheap smoke test (~$5)
- If results look promising, run full GSM8K
- Use `gpt-4o-mini` instead of `gpt-3.5-turbo` for everything except the exact replication (Phase 1A-C) — it's cheaper AND better
- Batch API for 50% discount on non-time-sensitive runs

**With batch API + mini for Phase 2+: Total could drop to $80-120**

---

## Timeline

| Day | Phase | What | Gate |
|-----|-------|------|------|
| 1 | Setup | Fork repos, download benchmarks, verify | Both codebases run |
| 2-3 | Baselines | Reproduce Du et al. results | Match ±2% |
| 4 | Integration | Build ARLM wrapper (the novel code) | Clean integration |
| 5-6 | Core experiment | Run scaling curves on GSM8K | Accuracy keeps climbing |
| 7-8 | Ablations | Component isolation, heterogeneous | Super-additivity shown |
| 9-10 | Write-up | Results section, figures, analysis | Paper ready for v2 |

**Calendar time: ~2 weeks** at Jeremy's ~2hr/day budget (mostly waiting for API runs)

Most of the human time is Day 1 (setup) and Day 4 (integration design). API runs can batch overnight.

---

## Success Criteria

### Minimum viable result (publishes):
- ARLM at 4+ rounds outperforms Du et al.'s 2-round debate by ≥3% on GSM8K
- Standard debate fails or degrades at 4+ rounds (context overflow)
- Clear scaling curve showing accuracy vs. rounds

### Strong result (high-impact):
- Super-additivity: ARLM > RLM_alone + Debate_alone
- Accuracy continues climbing to 6-8 rounds without saturation
- Heterogeneous models show additional gains

### If it fails:
- If ARLM doesn't improve over 2-round debate → RLM overhead isn't worth it for short-context tasks
- If accuracy plateaus at 4 rounds → debate has diminishing returns regardless of context
- Either outcome is still publishable as a negative result with the scaling analysis

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| GPT-3.5-turbo retired before we run | Low (still available) | Use gpt-4o-mini as backup, note in paper |
| Can't reproduce Du et al. baseline | Medium | Their code is public; debug before proceeding |
| RLM + debate integration is complex | Medium | Start with rlm-minimal (simple), not full rlm |
| Debate gains plateau at 3-4 rounds | Medium | This IS a finding — publish the scaling curve |
| Cost exceeds budget | Low | Smoke test on arithmetic first; batch API |
| Someone publishes ARLM first | Low-Medium | Move fast; GitHub timestamps our priority |

---

## Key Decisions Needed from Jeremy

1. **Budget approval:** ~$160-230 total (or ~$80-120 with batch optimization)
2. **Public repo name:** `adversarial-rlm`? `arlm`? `arlm-experiments`?
3. **Last name for paper** (still needed from concept paper)
4. **OpenAI API key** — do we have one, or need to set up billing?
5. **Timeline priority:** Start this week, or after RooftopSignal sprint?

---

*This plan is designed to be executable by sub-agents with minimal human intervention. Jeremy approves the plan + budget, we fork and run.*
