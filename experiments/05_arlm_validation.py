#!/usr/bin/env python3
"""ARLM Validation: 3-config comparison on GSM8K with gpt-4.1-nano.

Configs:
  1. Solo — single LLM call, no debate
  2. Standard debate — 3 agents, 2 rounds, plain text (Du et al.)
  3. ARLM — 3 agents, 2 rounds, persistent shared REPL

Runs the same N problems through all three. Reports accuracy + timing.
"""

import json
import os
import sys
import time
from collections import Counter
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv("/home/nootencorp/nootencorp-agent-worktrees/rsi-runner/.env")

from openai import OpenAI
from arlm.arlm_debate import (
    StandardDebate, ARLMDebate,
    extract_answer, extract_gsm8k_answer, answers_match,
)

MODEL = os.environ.get("MODEL", "gpt-3.5-turbo")
N_AGENTS = 3
N_ROUNDS = 2
N_PROBLEMS = int(os.environ.get("N_PROBLEMS", 20))
CONFIGS = os.environ.get("CONFIGS", "solo,standard_debate,arlm").split(",")

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_gsm8k(n, offset=0):
    """Load n problems from GSM8K starting at offset.
    If offset < 0, load the last n problems."""
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "benchmarks", "gsm8k", "test.jsonl",
    )
    with open(path) as f:
        all_problems = [json.loads(line) for line in f]
    if offset < 0:
        return all_problems[-n:]
    return all_problems[offset:offset + n]


def run_solo(question: str, api_key: str) -> dict:
    """Single LLM call — no debate, no REPL."""
    client = OpenAI(api_key=api_key, timeout=120)
    t0 = time.time()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": (
                f"Solve this math problem step by step. "
                f"Put your final numerical answer in \\boxed{{}}.\n\n{question}"
            ),
        }],
        temperature=0.7,
    )
    wall = time.time() - t0
    text = resp.choices[0].message.content or ""
    return {"answer_text": text, "wall_time": wall}


def majority_vote(answers: list) -> Optional[str]:
    """Extract and majority-vote from a list of answer texts."""
    extracted = [extract_answer(a) for a in answers]
    valid = [e for e in extracted if e is not None]
    if not valid:
        return None
    return Counter(valid).most_common(1)[0][0]


def run_config(config_name, problems, api_key):
    """Run one config across all problems. Returns list of result dicts."""
    results = []
    correct = 0

    for idx, prob in enumerate(problems):
        question = prob["question"]
        gold = extract_gsm8k_answer(prob["answer"])

        try:
            if config_name == "solo":
                res = run_solo(question, api_key)
                pred = extract_answer(res["answer_text"])
                wall = res["wall_time"]

            elif config_name == "standard_debate":
                debate = StandardDebate(
                    model=MODEL, api_key=api_key,
                    n_agents=N_AGENTS, n_rounds=N_ROUNDS,
                )
                res = debate.run(question)
                pred = majority_vote(res["final_answers"])
                wall = res["wall_time"]

            elif config_name == "arlm":
                debate = ARLMDebate(
                    model=MODEL, api_key=api_key,
                    n_agents=N_AGENTS, n_rounds=N_ROUNDS,
                    context_strategy="rlm",
                    early_exit=False,
                    max_repl_iterations=5,
                )
                res = debate.run(question)
                pred = majority_vote(res["final_answers"])
                wall = res["wall_time"]

            is_correct = answers_match(pred, gold)
            if is_correct:
                correct += 1

            result = {
                "problem_idx": idx,
                "config": config_name,
                "gold": gold,
                "predicted": pred,
                "correct": is_correct,
                "wall_time": wall,
            }
            results.append(result)

            status = "✓" if is_correct else "✗"
            print(f"  [{config_name}] {idx+1}/{len(problems)} {status} "
                  f"pred={pred} gold={gold} ({wall:.1f}s)")

        except Exception as e:
            print(f"  [{config_name}] {idx+1}/{len(problems)} ERROR: {e}")
            results.append({
                "problem_idx": idx,
                "config": config_name,
                "gold": gold,
                "predicted": None,
                "correct": False,
                "wall_time": 0,
                "error": str(e),
            })

    accuracy = correct / len(problems) * 100
    total_time = sum(r["wall_time"] for r in results)
    print(f"\n  [{config_name}] RESULT: {correct}/{len(problems)} = {accuracy:.1f}% "
          f"({total_time:.0f}s total)\n")

    return results, accuracy


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    offset = int(os.environ.get("OFFSET", -1))  # -1 = last N problems
    problems = load_gsm8k(N_PROBLEMS, offset=offset)

    print(f"╔══════════════════════════════════════════════╗")
    print(f"║  ARLM Validation: 3-Config Comparison       ║")
    print(f"║  Model: {MODEL:<20s} Problems: {N_PROBLEMS:<4d} ║")
    print(f"║  Agents: {N_AGENTS}  Rounds: {N_ROUNDS}                       ║")
    print(f"╚══════════════════════════════════════════════╝")
    print()

    all_results = {}
    accuracies = {}

    for ci, config_name in enumerate(CONFIGS):
        print(f"━━━ Config {ci+1}/{len(CONFIGS)}: {config_name} ━━━")
        res, acc = run_config(config_name, problems, api_key)
        all_results[config_name] = {"results": res, "accuracy": acc}
        accuracies[config_name] = acc

    # Summary
    print("╔══════════════════════════════════════════════╗")
    print("║              FINAL RESULTS                   ║")
    print("╠══════════════════════════════════════════════╣")
    for name, acc in accuracies.items():
        print(f"║  {name:<22s} {acc:5.1f}%                ║")
    print("╠══════════════════════════════════════════════╣")
    if "solo" in accuracies and "standard_debate" in accuracies:
        d = accuracies["standard_debate"] - accuracies["solo"]
        print(f"║  Debate vs Solo:       {d:+5.1f}pp                ║")
    if "solo" in accuracies and "arlm" in accuracies:
        d = accuracies["arlm"] - accuracies["solo"]
        print(f"║  ARLM vs Solo:         {d:+5.1f}pp                ║")
    if "standard_debate" in accuracies and "arlm" in accuracies:
        d = accuracies["arlm"] - accuracies["standard_debate"]
        print(f"║  ARLM vs Debate:       {d:+5.1f}pp                ║")
    print("╚══════════════════════════════════════════════╝")

    # Save results
    outpath = os.path.join(RESULTS_DIR, f"gsm8k_{N_PROBLEMS}p_{MODEL}.json")
    with open(outpath, "w") as f:
        json.dump({
            "model": MODEL,
            "n_problems": N_PROBLEMS,
            "n_agents": N_AGENTS,
            "n_rounds": N_ROUNDS,
            "summary": accuracies,
            "details": all_results,
        }, f, indent=2)
    print(f"\nResults saved to {outpath}")


if __name__ == "__main__":
    main()
