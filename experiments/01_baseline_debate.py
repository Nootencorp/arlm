#!/usr/bin/env python3
"""
Experiment 1: Reproduce Du et al.'s multiagent debate baseline.

Runs StandardDebate on GSM8K test set and reports accuracy.
Expected result: ~89% with 3 agents, 2 rounds, gpt-3.5-turbo.

Usage:
    # Full run (paid API)
    python experiments/01_baseline_debate.py \\
        --model gpt-3.5-turbo --n-problems 1319

    # Quick test with free model
    python experiments/01_baseline_debate.py \\
        --model meta-llama/llama-3.3-70b-instruct:free \\
        --base-url https://openrouter.ai/api/v1 \\
        --api-key-env OPENROUTER_API_KEY \\
        --n-problems 5
"""

import argparse
import json
import os
import sys
import time

# Ensure the project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arlm.arlm_debate import (
    StandardDebate,
    extract_answer,
    extract_gsm8k_answer,
    answers_match,
)


# ---------------------------------------------------------------------------
# GSM8K loader
# ---------------------------------------------------------------------------

def load_gsm8k(path: str, limit: int = None):
    """
    Load GSM8K test set from a JSONL file.

    Each line has keys: ``question``, ``answer``.
    Returns a list of dicts with ``question`` and ``gold_answer`` (numeric).
    """
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
                "raw_answer": item["answer"],
            })
            if limit and len(problems) >= limit:
                break
    return problems


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Baseline multi-agent debate on GSM8K (Du et al. 2023).",
    )
    parser.add_argument("--model", default="gpt-3.5-turbo",
                        help="Model name (default: gpt-3.5-turbo)")
    parser.add_argument("--n-agents", type=int, default=3,
                        help="Number of debate agents (default: 3)")
    parser.add_argument("--n-rounds", type=int, default=2,
                        help="Number of debate rounds (default: 2)")
    parser.add_argument("--n-problems", type=int, default=100,
                        help="How many GSM8K problems to evaluate (default: 100)")
    parser.add_argument("--base-url", default=None,
                        help="API base URL (e.g. https://openrouter.ai/api/v1)")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY",
                        help="Env var holding the API key (default: OPENAI_API_KEY)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--output", default="results/01_baseline.jsonl",
                        help="Output JSONL path (default: results/01_baseline.jsonl)")
    parser.add_argument("--gsm-path", default="benchmarks/gsm8k/test.jsonl",
                        help="Path to GSM8K test JSONL")
    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gsm_path = os.path.join(project_root, args.gsm_path) if not os.path.isabs(args.gsm_path) else args.gsm_path
    output_path = os.path.join(project_root, args.output) if not os.path.isabs(args.output) else args.output

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load GSM8K
    print(f"Loading GSM8K from {gsm_path} ...")
    problems = load_gsm8k(gsm_path, limit=args.n_problems)
    print(f"  Loaded {len(problems)} problems")

    # API key
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        print(f"ERROR: Environment variable {args.api_key_env} is not set.")
        sys.exit(1)

    # Build debater
    debater = StandardDebate(
        model=args.model,
        n_agents=args.n_agents,
        n_rounds=args.n_rounds,
        api_key=api_key,
        base_url=args.base_url,
        temperature=args.temperature,
    )

    # Run
    correct = 0
    total = 0
    total_tokens = 0
    total_time = 0.0

    print(f"\nRunning StandardDebate  model={args.model}  agents={args.n_agents}  rounds={args.n_rounds}")
    print(f"{'='*70}")

    with open(output_path, "w") as fout:
        for idx, prob in enumerate(problems):
            q = prob["question"]
            gold = prob["gold_answer"]

            print(f"\n[{idx+1}/{len(problems)}] {q[:80]}...")

            try:
                result = debater.run(q)
            except Exception as e:
                print(f"  ERROR: {e}")
                record = {
                    "idx": idx,
                    "question": q,
                    "gold": gold,
                    "error": str(e),
                    "correct": False,
                }
                fout.write(json.dumps(record) + "\n")
                fout.flush()
                total += 1
                continue

            # Majority vote across agents' final answers
            agent_answers = []
            for resp in result["final_answers"]:
                ans = extract_answer(resp)
                agent_answers.append(ans)

            # Majority vote
            vote_counts: dict = {}
            for a in agent_answers:
                if a is not None:
                    vote_counts[a] = vote_counts.get(a, 0) + 1
            if vote_counts:
                pred = max(vote_counts, key=vote_counts.get)
            else:
                pred = None

            is_correct = answers_match(pred, gold)
            correct += int(is_correct)
            total += 1
            total_tokens += result["token_count"]
            total_time += result["wall_time"]

            accuracy = correct / total if total > 0 else 0
            print(f"  pred={pred}  gold={gold}  {'✓' if is_correct else '✗'}  "
                  f"running_acc={accuracy:.1%}  tokens={result['token_count']}  "
                  f"time={result['wall_time']:.1f}s")

            # Save result
            record = {
                "idx": idx,
                "question": q,
                "gold": gold,
                "pred": pred,
                "agent_answers": agent_answers,
                "correct": is_correct,
                "token_count": result["token_count"],
                "wall_time": result["wall_time"],
            }
            fout.write(json.dumps(record) + "\n")
            fout.flush()

    # Summary
    accuracy = correct / total if total > 0 else 0
    print(f"\n{'='*70}")
    print(f"RESULTS: {correct}/{total} = {accuracy:.1%} accuracy")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Total time:   {total_time:.1f}s")
    print(f"Results saved to {output_path}")

    # Also save a summary JSON
    summary_path = output_path.replace(".jsonl", "_summary.json")
    summary = {
        "model": args.model,
        "n_agents": args.n_agents,
        "n_rounds": args.n_rounds,
        "n_problems": total,
        "correct": correct,
        "accuracy": accuracy,
        "total_tokens": total_tokens,
        "total_time": total_time,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
