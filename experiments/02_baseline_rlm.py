#!/usr/bin/env python3
"""
Experiment 2: RLM-only baseline (no debate).

Tests whether RLM improves single-agent GSM8K accuracy.
Compares against raw single-agent and debate baselines.

This isolates the contribution of RLM wrapping for ablation studies.
For GSM8K (short problems), RLM may not provide much benefit, but we need
this data point to test super-additivity: ARLM > RLM_alone + Debate_alone.

Usage:
    # Raw LLM mode (no RLM)
    python experiments/02_baseline_rlm.py --mode raw --n-problems 100
    
    # RLM-wrapped mode
    python experiments/02_baseline_rlm.py --mode rlm --n-problems 100
    
    # With custom model
    python experiments/02_baseline_rlm.py --mode rlm --model gpt-4-turbo --n-problems 100
"""

import argparse
import json
import os
import sys
import time

# Ensure the project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arlm.arlm_debate import extract_answer, extract_gsm8k_answer, answers_match

# Add rlm_src to sys.path for RLM imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rlm_src"))
from rlm.rlm_repl import RLM_REPL

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not found. Install with: pip install openai>=1.0", file=sys.stderr)
    sys.exit(1)


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
# Raw LLM solver
# ---------------------------------------------------------------------------

def solve_raw(question: str, client: OpenAI, model: str, temperature: float = 0.7) -> dict:
    """
    Solve a math problem using raw LLM (no RLM, no debate).
    
    Returns:
        dict with keys: answer, raw_response, token_count, wall_time
    """
    system_prompt = (
        "You are a helpful assistant that solves math word problems step-by-step. "
        "Show your work and provide the final numerical answer."
    )
    
    start = time.time()
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            temperature=temperature,
        )
        
        wall_time = time.time() - start
        
        raw_response = response.choices[0].message.content
        answer = extract_answer(raw_response)
        token_count = response.usage.total_tokens
        
        return {
            "answer": answer,
            "raw_response": raw_response,
            "token_count": token_count,
            "wall_time": wall_time,
        }
    
    except Exception as e:
        wall_time = time.time() - start
        return {
            "answer": None,
            "raw_response": None,
            "token_count": 0,
            "wall_time": wall_time,
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# RLM solver
# ---------------------------------------------------------------------------

def solve_rlm(question: str, api_key: str, model: str, temperature: float = 0.7) -> dict:
    """
    Solve a math problem using RLM (recursive language model with REPL).
    
    The RLM wraps the problem as context and allows the model to programmatically
    work through it using a REPL environment.
    
    Returns:
        dict with keys: answer, raw_response, token_count, wall_time
    """
    start = time.time()
    
    try:
        # Initialize RLM with REPL
        rlm = RLM_REPL(
            api_key=api_key,
            model=model,
            recursive_model=model,
            max_iterations=20,
            enable_logging=False,
        )
        
        # Setup context with the problem
        # The problem becomes part of the context that RLM can reference
        context = f"Math Problem:\n{question}\n\nSolve this problem step by step and provide the final numerical answer."
        
        query = "What is the answer to this math problem? Show your work and provide the final numerical answer."
        
        # Run RLM completion
        raw_response = rlm.completion(context=context, query=query)
        
        wall_time = time.time() - start
        
        # Extract answer
        answer = extract_answer(raw_response)
        
        # Token counting for RLM is more complex (multiple LLM calls)
        # For now, we estimate based on the number of REPL iterations
        # This is a rough approximation
        token_count = 0  # RLM doesn't expose total tokens easily
        
        return {
            "answer": answer,
            "raw_response": raw_response,
            "token_count": token_count,
            "wall_time": wall_time,
        }
    
    except Exception as e:
        wall_time = time.time() - start
        return {
            "answer": None,
            "raw_response": None,
            "token_count": 0,
            "wall_time": wall_time,
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="RLM-only baseline on GSM8K (ablation experiment).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Raw mode (no RLM)
    python experiments/02_baseline_rlm.py --mode raw --n-problems 100
    
    # RLM mode
    python experiments/02_baseline_rlm.py --mode rlm --n-problems 100
        """
    )
    
    parser.add_argument("--mode", choices=["raw", "rlm"], default="raw",
                        help="Solver mode: 'raw' (just LLM) or 'rlm' (RLM-wrapped)")
    parser.add_argument("--model", default="gpt-3.5-turbo",
                        help="Model name (default: gpt-3.5-turbo)")
    parser.add_argument("--n-problems", type=int, default=100,
                        help="How many GSM8K problems to evaluate (default: 100)")
    parser.add_argument("--base-url", default=None,
                        help="API base URL (e.g. https://openrouter.ai/api/v1)")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY",
                        help="Env var holding the API key (default: OPENAI_API_KEY)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--output", default=None,
                        help="Output JSONL path (default: results/phase1_{mode}_{n}.jsonl)")
    parser.add_argument("--gsm-path", default="benchmarks/gsm8k/test.jsonl",
                        help="Path to GSM8K test JSONL")
    
    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gsm_path = os.path.join(project_root, args.gsm_path) if not os.path.isabs(args.gsm_path) else args.gsm_path
    
    if args.output is None:
        args.output = f"results/phase1_{args.mode}_{args.n_problems}.jsonl"
    
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

    # Initialize OpenAI client for raw mode
    client = None
    if args.mode == "raw":
        client = OpenAI(api_key=api_key, base_url=args.base_url)

    # Run
    correct = 0
    total = 0
    total_tokens = 0
    total_time = 0.0

    print(f"\nRunning {args.mode.upper()} mode  model={args.model}  n={args.n_problems}")
    print(f"{'='*70}")

    with open(output_path, "w") as fout:
        for idx, prob in enumerate(problems):
            q = prob["question"]
            gold = prob["gold_answer"]

            print(f"\n[{idx+1}/{len(problems)}] {q[:80]}...")

            try:
                if args.mode == "raw":
                    result = solve_raw(q, client, args.model, args.temperature)
                else:  # rlm
                    result = solve_rlm(q, api_key, args.model, args.temperature)
                
                if "error" in result:
                    print(f"  ERROR: {result['error']}")
                    record = {
                        "idx": idx,
                        "question": q,
                        "gold": gold,
                        "error": result["error"],
                        "correct": False,
                        "token_count": result["token_count"],
                        "wall_time": result["wall_time"],
                    }
                    fout.write(json.dumps(record) + "\n")
                    fout.flush()
                    continue

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
                continue

            pred = result["answer"]
            is_correct = answers_match(pred, gold)

            print(f"  Gold={gold}  Pred={pred}  Correct={is_correct}")
            print(f"  Tokens={result['token_count']}  Time={result['wall_time']:.2f}s")

            correct += is_correct
            total += 1
            total_tokens += result["token_count"]
            total_time += result["wall_time"]

            record = {
                "idx": idx,
                "question": q,
                "gold": gold,
                "pred": pred,
                "raw_response": result["raw_response"],
                "correct": is_correct,
                "token_count": result["token_count"],
                "wall_time": result["wall_time"],
            }
            fout.write(json.dumps(record) + "\n")
            fout.flush()

    # Final summary
    accuracy = correct / total if total > 0 else 0.0
    avg_tokens = total_tokens / total if total > 0 else 0.0
    avg_time = total_time / total if total > 0 else 0.0

    print(f"\n{'='*70}")
    print(f"FINAL RESULTS ({args.mode.upper()} mode)")
    print(f"  Accuracy: {correct}/{total} = {accuracy:.1%}")
    print(f"  Avg tokens: {avg_tokens:.0f}")
    print(f"  Avg time: {avg_time:.2f}s")
    print(f"\nResults written to: {output_path}")

    # Write summary
    summary_path = output_path.replace(".jsonl", "_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "mode": args.mode,
            "model": args.model,
            "n_problems": total,
            "accuracy": accuracy,
            "correct": correct,
            "avg_tokens": avg_tokens,
            "avg_time": avg_time,
        }, f, indent=2)
    print(f"Summary written to: {summary_path}")


if __name__ == '__main__':
    main()
