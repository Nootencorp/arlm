#!/usr/bin/env python3
"""ARLM Validation on MATH dataset (Level 5 — hardest).

3-config comparison: Solo vs Standard Debate vs ARLM.
Uses sympy for LaTeX answer equivalence where possible.
"""

import json
import os
import re
import sys
import time
from collections import Counter
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv("/home/nootencorp/nootencorp-agent-worktrees/rsi-runner/.env")

from openai import OpenAI
from arlm.arlm_debate import StandardDebate, ARLMDebate

MODEL = os.environ.get("MODEL", "gpt-3.5-turbo")
N_AGENTS = 3
N_ROUNDS = 2
N_PROBLEMS = int(os.environ.get("N_PROBLEMS", 20))
LEVEL = int(os.environ.get("MATH_LEVEL", 5))
CONFIGS = os.environ.get("CONFIGS", "solo,standard_debate,arlm").split(",")

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_math(n, level=5):
    """Load n problems from MATH dataset at given level."""
    if level == 5:
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "benchmarks", "math500", "level5.jsonl",
        )
    else:
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "benchmarks", "math500", "all.jsonl",
        )
    with open(path) as f:
        all_probs = [json.loads(line) for line in f]
        if level != 5:
            all_probs = [p for p in all_probs if p["level"] == level]
    return all_probs[:n]


def extract_boxed(text: str) -> Optional[str]:
    """Extract the last \\boxed{...} from text, handling nested braces."""
    # Find all \boxed occurrences
    results = []
    i = 0
    while i < len(text):
        idx = text.find("\\boxed{", i)
        if idx == -1:
            break
        # Find matching closing brace
        depth = 0
        start = idx + 7  # after \boxed{
        j = start
        while j < len(text):
            if text[j] == '{':
                depth += 1
            elif text[j] == '}':
                if depth == 0:
                    results.append(text[start:j])
                    break
                depth -= 1
            j += 1
        i = j + 1
    return results[-1] if results else None


def normalize_answer(ans: str) -> str:
    """Normalize a math answer string for comparison."""
    if ans is None:
        return ""
    ans = ans.strip()
    # Remove \text{}, \mathrm{}, etc.
    ans = re.sub(r'\\(?:text|mathrm|textbf)\{([^}]*)\}', r'\1', ans)
    # Remove \left, \right
    ans = ans.replace("\\left", "").replace("\\right", "")
    # Remove leading variable assignment (e.g., "w = 6 - 5i" -> "6 - 5i", "x=5" -> "5")
    ans = re.sub(r'^[a-zA-Z]\s*=\s*', '', ans)
    # Remove spaces
    ans = ans.replace(" ", "")
    # Remove trailing period
    ans = ans.rstrip(".")
    # Normalize common forms
    ans = ans.replace("\\dfrac", "\\frac")
    ans = ans.replace("\\tfrac", "\\frac")
    return ans


def sort_set_answer(ans: str) -> str:
    """Sort comma-separated values for set comparison (e.g. '1,-2' == '-2,1')."""
    parts = [p.strip() for p in ans.split(",")]
    try:
        # Try numeric sort
        parts.sort(key=float)
    except ValueError:
        parts.sort()
    return ",".join(parts)


def answers_match(pred: Optional[str], gold: Optional[str]) -> bool:
    """Compare two MATH answers with normalization."""
    if pred is None or gold is None:
        return False
    
    np = normalize_answer(pred)
    ng = normalize_answer(gold)
    
    # Direct string match after normalization
    if np == ng:
        return True
    
    # Try set comparison (comma-separated values, order-independent)
    if "," in np or "," in ng:
        if sort_set_answer(np) == sort_set_answer(ng):
            return True
    
    # Try numeric comparison
    try:
        def eval_expr(s):
            s = s.replace("\\frac{", "(").replace("}{", ")/(").replace("}", ")")
            s = s.replace("\\pi", "*3.14159265358979")
            s = re.sub(r'\\sqrt\{([^}]+)\}', r'((\1)**0.5)', s)
            return float(eval(s))
        
        if abs(eval_expr(np) - eval_expr(ng)) < 1e-6:
            return True
    except:
        pass
    
    return False


def run_solo(question: str, api_key: str) -> dict:
    """Single LLM call."""
    client = OpenAI(api_key=api_key, timeout=120)
    t0 = time.time()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": (
                f"Solve this math problem step by step. "
                f"Put your final answer in \\boxed{{}}.\n\n{question}"
            ),
        }],
        temperature=0.7,
    )
    wall = time.time() - t0
    text = resp.choices[0].message.content or ""
    return {"answer_text": text, "wall_time": wall}


def majority_vote(answers: list) -> Optional[str]:
    """Extract and majority-vote from answer texts."""
    extracted = [extract_boxed(a) for a in answers]
    valid = [e for e in extracted if e is not None]
    if not valid:
        return None
    # Normalize for voting
    normed = [normalize_answer(e) for e in valid]
    most_common = Counter(normed).most_common(1)[0][0]
    # Return the original (unnormalized) form of the winner
    for e, n in zip(valid, normed):
        if n == most_common:
            return e
    return valid[0]


def run_config(config_name, problems, api_key):
    """Run one config across all problems."""
    results = []
    correct = 0

    for idx, prob in enumerate(problems):
        question = prob["problem"]
        gold = prob["answer"]

        try:
            if config_name == "solo":
                res = run_solo(question, api_key)
                pred = extract_boxed(res["answer_text"])
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
                "subject": prob.get("subject", ""),
            }
            results.append(result)

            status = "✓" if is_correct else "✗"
            print(f"  [{config_name}] {idx+1}/{len(problems)} {status} "
                  f"pred={pred} gold={gold} ({wall:.1f}s) [{prob.get('subject','')}]")

        except Exception as e:
            import traceback
            traceback.print_exc()
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

    accuracy = correct / len(problems) * 100 if problems else 0
    total_time = sum(r["wall_time"] for r in results)
    print(f"\n  [{config_name}] RESULT: {correct}/{len(problems)} = {accuracy:.1f}% "
          f"({total_time:.0f}s total)\n")

    return results, accuracy


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    problems = load_math(N_PROBLEMS, level=LEVEL)

    print(f"╔══════════════════════════════════════════════════╗")
    print(f"║  ARLM Validation: MATH Level {LEVEL} ({len(problems)} problems) ║")
    print(f"║  Model: {MODEL:<20s} Configs: {','.join(CONFIGS):<10s}║")
    print(f"║  Agents: {N_AGENTS}  Rounds: {N_ROUNDS}                          ║")
    print(f"╚══════════════════════════════════════════════════╝")
    print()

    all_results = {}
    accuracies = {}

    for ci, config_name in enumerate(CONFIGS):
        print(f"━━━ Config {ci+1}/{len(CONFIGS)}: {config_name} ━━━")
        res, acc = run_config(config_name, problems, api_key)
        all_results[config_name] = {"results": res, "accuracy": acc}
        accuracies[config_name] = acc

    # Summary
    print("╔══════════════════════════════════════════════════╗")
    print("║              FINAL RESULTS                       ║")
    print("╠══════════════════════════════════════════════════╣")
    for name, acc in accuracies.items():
        print(f"║  {name:<22s} {acc:5.1f}%                    ║")
    print("╠══════════════════════════════════════════════════╣")
    if "solo" in accuracies and "standard_debate" in accuracies:
        d = accuracies["standard_debate"] - accuracies["solo"]
        print(f"║  Debate vs Solo:       {d:+5.1f}pp                   ║")
    if "solo" in accuracies and "arlm" in accuracies:
        d = accuracies["arlm"] - accuracies["solo"]
        print(f"║  ARLM vs Solo:         {d:+5.1f}pp                   ║")
    if "standard_debate" in accuracies and "arlm" in accuracies:
        d = accuracies["arlm"] - accuracies["standard_debate"]
        print(f"║  ARLM vs Debate:       {d:+5.1f}pp                   ║")
    print("╚══════════════════════════════════════════════════╝")

    # Save
    outpath = os.path.join(RESULTS_DIR, f"math_L{LEVEL}_{N_PROBLEMS}p_{MODEL}.json")
    with open(outpath, "w") as f:
        json.dump({
            "model": MODEL,
            "n_problems": N_PROBLEMS,
            "level": LEVEL,
            "n_agents": N_AGENTS,
            "n_rounds": N_ROUNDS,
            "summary": accuracies,
            "details": all_results,
        }, f, indent=2)
    print(f"\nResults saved to {outpath}")


if __name__ == "__main__":
    main()
