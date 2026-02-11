#!/usr/bin/env python3
"""
Experiment 3: ARLM scaling curve.

Runs debate at multiple round counts with different context strategies.
Compares standard debate vs ARLM as rounds increase.

Configurations:
    - Standard debate at 2, 4 rounds
    - ARLM (summary) at 2, 4, 6, 8 rounds
    - ARLM (rlm) at 2, 4, 6, 8 rounds (optional — requires RLM deps)

Output: per-config JSONL + comparison table on stdout.

Usage:
    # Quick test with free model
    python experiments/03_arlm_scaled.py \\
        --model meta-llama/llama-3.3-70b-instruct:free \\
        --base-url https://openrouter.ai/api/v1 \\
        --api-key-env OPENROUTER_API_KEY \\
        --n-problems 5

    # Full run
    python experiments/03_arlm_scaled.py \\
        --model gpt-3.5-turbo --n-problems 100
"""

import argparse
import json
import os
import sys
import time
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arlm.arlm_debate import (
    StandardDebate,
    ARLMDebate,
    extract_answer,
    extract_gsm8k_answer,
    answers_match,
)


# ---------------------------------------------------------------------------
# GSM8K loader (shared with 01)
# ---------------------------------------------------------------------------

def load_gsm8k(path: str, limit: int = None):
    problems = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            gold = extract_gsm8k_answer(item["answer"])
            problems.append({
                "question": item["question"],
                "gold_answer": gold,
            })
            if limit and len(problems) >= limit:
                break
    return problems


# ---------------------------------------------------------------------------
# Run one configuration
# ---------------------------------------------------------------------------

def evaluate_config(
    name: str,
    debater,
    problems: List[dict],
    output_path: str,
) -> dict:
    """
    Run *debater* on *problems*, write JSONL, return summary dict.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    correct = 0
    total = 0
    total_tokens = 0
    total_time = 0.0

    print(f"\n{'─'*60}")
    print(f"Config: {name}")
    print(f"{'─'*60}")

    with open(output_path, "w") as fout:
        for idx, prob in enumerate(problems):
            q = prob["question"]
            gold = prob["gold_answer"]

            print(f"  [{idx+1}/{len(problems)}] {q[:60]}...", end="  ")

            try:
                result = debater.run(q)
            except Exception as e:
                print(f"ERROR: {e}")
                fout.write(json.dumps({
                    "idx": idx, "question": q, "gold": gold,
                    "error": str(e), "correct": False,
                }) + "\n")
                fout.flush()
                total += 1
                continue

            # Majority vote
            agent_answers = [extract_answer(r) for r in result["final_answers"]]
            vote_counts: dict = {}
            for a in agent_answers:
                if a is not None:
                    vote_counts[a] = vote_counts.get(a, 0) + 1
            pred = max(vote_counts, key=vote_counts.get) if vote_counts else None

            is_correct = answers_match(pred, gold)
            correct += int(is_correct)
            total += 1
            total_tokens += result.get("token_count", 0)
            total_time += result.get("wall_time", 0)

            print(f"pred={pred} gold={gold} {'✓' if is_correct else '✗'}")

            fout.write(json.dumps({
                "idx": idx, "question": q, "gold": gold,
                "pred": pred, "agent_answers": agent_answers,
                "correct": is_correct,
                "token_count": result.get("token_count", 0),
                "wall_time": result.get("wall_time", 0),
            }) + "\n")
            fout.flush()

    accuracy = correct / total if total > 0 else 0.0
    return {
        "name": name,
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "total_tokens": total_tokens,
        "total_time": total_time,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ARLM scaling curve: compare standard debate vs ARLM at increasing rounds.",
    )
    parser.add_argument("--model", default="gpt-3.5-turbo")
    parser.add_argument("--n-agents", type=int, default=3)
    parser.add_argument("--n-problems", type=int, default=100)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--output", default="results/03_arlm_scaled",
                        help="Output directory prefix (default: results/03_arlm_scaled)")
    parser.add_argument("--gsm-path", default="benchmarks/gsm8k/test.jsonl")
    parser.add_argument("--standard-rounds", nargs="+", type=int,
                        default=[2, 4],
                        help="Round counts for standard debate (default: 2 4)")
    parser.add_argument("--arlm-rounds", nargs="+", type=int,
                        default=[2, 4, 6, 8],
                        help="Round counts for ARLM strategies (default: 2 4 6 8)")
    parser.add_argument("--skip-rlm", action="store_true",
                        help="Skip the RLM strategy (requires RLM deps)")
    args = parser.parse_args()

    # Resolve paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gsm_path = os.path.join(project_root, args.gsm_path) if not os.path.isabs(args.gsm_path) else args.gsm_path
    output_dir = os.path.join(project_root, args.output) if not os.path.isabs(args.output) else args.output
    os.makedirs(output_dir, exist_ok=True)

    # API key
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        print(f"ERROR: Environment variable {args.api_key_env} is not set.")
        sys.exit(1)

    # Load problems
    print(f"Loading GSM8K from {gsm_path} ...")
    problems = load_gsm8k(gsm_path, limit=args.n_problems)
    print(f"  Loaded {len(problems)} problems")

    # Shared kwargs
    common = dict(
        model=args.model,
        n_agents=args.n_agents,
        api_key=api_key,
        base_url=args.base_url,
        temperature=args.temperature,
    )

    summaries: List[dict] = []

    # 1. Standard debate at various rounds
    for n_rounds in args.standard_rounds:
        name = f"standard_r{n_rounds}"
        debater = StandardDebate(n_rounds=n_rounds, **common)
        out = os.path.join(output_dir, f"{name}.jsonl")
        s = evaluate_config(name, debater, problems, out)
        summaries.append(s)

    # 2. ARLM (summary) at various rounds
    for n_rounds in args.arlm_rounds:
        name = f"arlm_summary_r{n_rounds}"
        debater = ARLMDebate(
            n_rounds=n_rounds,
            context_strategy="summary",
            **common,
        )
        out = os.path.join(output_dir, f"{name}.jsonl")
        s = evaluate_config(name, debater, problems, out)
        summaries.append(s)

    # 3. ARLM (rlm) at various rounds (optional)
    if not args.skip_rlm:
        for n_rounds in args.arlm_rounds:
            name = f"arlm_rlm_r{n_rounds}"
            try:
                debater = ARLMDebate(
                    n_rounds=n_rounds,
                    context_strategy="rlm",
                    **common,
                )
                out = os.path.join(output_dir, f"{name}.jsonl")
                s = evaluate_config(name, debater, problems, out)
                summaries.append(s)
            except Exception as e:
                print(f"\n  Skipping {name}: {e}")
                summaries.append({
                    "name": name,
                    "correct": 0, "total": 0,
                    "accuracy": 0.0,
                    "total_tokens": 0,
                    "total_time": 0.0,
                    "error": str(e),
                })

    # Print comparison table
    print(f"\n\n{'='*80}")
    print("SCALING CURVE — COMPARISON TABLE")
    print(f"{'='*80}")
    print(f"{'Config':<25} {'Correct':>8} {'Total':>6} {'Accuracy':>9} "
          f"{'Tokens':>10} {'Time (s)':>10}")
    print(f"{'─'*25} {'─'*8} {'─'*6} {'─'*9} {'─'*10} {'─'*10}")
    for s in summaries:
        print(f"{s['name']:<25} {s['correct']:>8} {s['total']:>6} "
              f"{s['accuracy']:>8.1%} {s['total_tokens']:>10,} "
              f"{s['total_time']:>10.1f}")
    print(f"{'='*80}")

    # Save summary JSON
    summary_path = os.path.join(output_dir, "comparison_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
