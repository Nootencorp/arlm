#!/usr/bin/env python3
"""
Experiment 4: Ablation studies.

Runs component isolation to test super-additivity:
  - Single agent (no debate, no RLM) 
  - Debate only (Du et al. standard, 2 rounds)
  - RLM only (no debate)
  - ARLM (debate + RLM, 4 rounds)
  
Also tests heterogeneous models (e.g., GPT-4 moderator with GPT-3.5 agents).

Usage:
    # Run all ablations with default settings
    python experiments/04_ablations.py --n-problems 100
    
    # Custom output directory
    python experiments/04_ablations.py --n-problems 100 --output-dir results/ablations
    
    # Test heterogeneous models
    python experiments/04_ablations.py --heterogeneous --n-problems 100
"""

import argparse
import json
import os
import sys
import time
from typing import List, Dict, Any

# Ensure the project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arlm.arlm_debate import (
    StandardDebate,
    ARLMDebate,
    extract_answer,
    extract_gsm8k_answer,
    answers_match,
)

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
    """Load GSM8K test set."""
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
# Ablation components
# ---------------------------------------------------------------------------

def run_single_agent(problems: List[Dict], client: OpenAI, model: str, temperature: float = 0.7) -> List[Dict]:
    """Run single agent (no debate, no RLM)."""
    results = []
    
    system_prompt = (
        "You are a helpful assistant that solves math word problems step-by-step. "
        "Show your work and provide the final numerical answer."
    )
    
    for idx, prob in enumerate(problems):
        q = prob["question"]
        gold = prob["gold_answer"]
        
        print(f"\n[Single Agent {idx+1}/{len(problems)}] {q[:60]}...")
        
        start = time.time()
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": q}
                ],
                temperature=temperature,
            )
            
            wall_time = time.time() - start
            raw_response = response.choices[0].message.content
            pred = extract_answer(raw_response)
            token_count = response.usage.total_tokens
            is_correct = answers_match(pred, gold)
            
            print(f"  Gold={gold}  Pred={pred}  Correct={is_correct}")
            
            results.append({
                "idx": idx,
                "question": q,
                "gold": gold,
                "pred": pred,
                "correct": is_correct,
                "token_count": token_count,
                "wall_time": wall_time,
            })
        
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "idx": idx,
                "question": q,
                "gold": gold,
                "error": str(e),
                "correct": False,
            })
    
    return results


def run_debate_only(problems: List[Dict], api_key: str, model: str, n_agents: int = 3, 
                    n_rounds: int = 2, temperature: float = 0.7, base_url: str = None) -> List[Dict]:
    """Run standard debate (no RLM)."""
    results = []
    
    debater = StandardDebate(
        model=model,
        n_agents=n_agents,
        n_rounds=n_rounds,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
    )
    
    for idx, prob in enumerate(problems):
        q = prob["question"]
        gold = prob["gold_answer"]
        
        print(f"\n[Debate Only {idx+1}/{len(problems)}] {q[:60]}...")
        
        try:
            result = debater.run(q)
            pred = result["answer"]
            is_correct = answers_match(pred, gold)
            
            print(f"  Gold={gold}  Pred={pred}  Correct={is_correct}")
            print(f"  Tokens={result['token_count']}  Time={result['wall_time']:.2f}s")
            
            results.append({
                "idx": idx,
                "question": q,
                "gold": gold,
                "pred": pred,
                "agent_answers": result.get("agent_answers", []),
                "correct": is_correct,
                "token_count": result["token_count"],
                "wall_time": result["wall_time"],
            })
        
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "idx": idx,
                "question": q,
                "gold": gold,
                "error": str(e),
                "correct": False,
            })
    
    return results


def run_rlm_only(problems: List[Dict], api_key: str, model: str, temperature: float = 0.7) -> List[Dict]:
    """Run RLM only (no debate)."""
    results = []
    
    for idx, prob in enumerate(problems):
        q = prob["question"]
        gold = prob["gold_answer"]
        
        print(f"\n[RLM Only {idx+1}/{len(problems)}] {q[:60]}...")
        
        start = time.time()
        
        try:
            rlm = RLM_REPL(
                api_key=api_key,
                model=model,
                recursive_model=model,
                max_iterations=20,
                enable_logging=False,
            )
            
            context = f"Math Problem:\n{q}\n\nSolve this problem step by step and provide the final numerical answer."
            query = "What is the answer to this math problem? Show your work and provide the final numerical answer."
            
            raw_response = rlm.completion(context=context, query=query)
            wall_time = time.time() - start
            
            pred = extract_answer(raw_response)
            is_correct = answers_match(pred, gold)
            
            print(f"  Gold={gold}  Pred={pred}  Correct={is_correct}")
            
            results.append({
                "idx": idx,
                "question": q,
                "gold": gold,
                "pred": pred,
                "correct": is_correct,
                "token_count": 0,  # RLM doesn't expose total tokens easily
                "wall_time": wall_time,
            })
        
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "idx": idx,
                "question": q,
                "gold": gold,
                "error": str(e),
                "correct": False,
            })
    
    return results


def run_arlm(problems: List[Dict], api_key: str, model: str, strategy: str = "summary",
             n_agents: int = 3, n_rounds: int = 4, temperature: float = 0.7, 
             base_url: str = None) -> List[Dict]:
    """Run ARLM (debate + RLM)."""
    results = []
    
    debater = ARLMDebate(
        model=model,
        n_agents=n_agents,
        n_rounds=n_rounds,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        strategy=strategy,
    )
    
    for idx, prob in enumerate(problems):
        q = prob["question"]
        gold = prob["gold_answer"]
        
        print(f"\n[ARLM {idx+1}/{len(problems)}] {q[:60]}...")
        
        try:
            result = debater.run(q)
            pred = result["answer"]
            is_correct = answers_match(pred, gold)
            
            print(f"  Gold={gold}  Pred={pred}  Correct={is_correct}")
            print(f"  Tokens={result['token_count']}  Time={result['wall_time']:.2f}s")
            
            results.append({
                "idx": idx,
                "question": q,
                "gold": gold,
                "pred": pred,
                "agent_answers": result.get("agent_answers", []),
                "correct": is_correct,
                "token_count": result["token_count"],
                "wall_time": result["wall_time"],
            })
        
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "idx": idx,
                "question": q,
                "gold": gold,
                "error": str(e),
                "correct": False,
            })
    
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Ablation study: isolate debate and RLM contributions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--model", default="gpt-3.5-turbo",
                        help="Primary model (default: gpt-3.5-turbo)")
    parser.add_argument("--n-problems", type=int, default=100,
                        help="Number of GSM8K problems (default: 100)")
    parser.add_argument("--base-url", default=None,
                        help="API base URL")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY",
                        help="Env var for API key (default: OPENAI_API_KEY)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--output-dir", default="results/ablations",
                        help="Output directory (default: results/ablations)")
    parser.add_argument("--gsm-path", default="benchmarks/gsm8k/test.jsonl",
                        help="Path to GSM8K test JSONL")
    parser.add_argument("--heterogeneous", action="store_true",
                        help="Test heterogeneous models (GPT-4 moderator, GPT-3.5 agents)")
    parser.add_argument("--components", nargs="+", 
                        choices=["single", "debate", "rlm", "arlm", "all"],
                        default=["all"],
                        help="Which components to run (default: all)")
    
    args = parser.parse_args()

    # Resolve paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gsm_path = os.path.join(project_root, args.gsm_path) if not os.path.isabs(args.gsm_path) else args.gsm_path
    output_dir = os.path.join(project_root, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Load GSM8K
    print(f"Loading GSM8K from {gsm_path} ...")
    problems = load_gsm8k(gsm_path, limit=args.n_problems)
    print(f"  Loaded {len(problems)} problems")

    # API key
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        print(f"ERROR: Environment variable {args.api_key_env} is not set.")
        sys.exit(1)

    # Determine which components to run
    components = args.components
    if "all" in components:
        components = ["single", "debate", "rlm", "arlm"]

    # Initialize OpenAI client
    client = OpenAI(api_key=api_key, base_url=args.base_url)

    print(f"\n{'='*70}")
    print(f"ABLATION STUDY")
    print(f"Model: {args.model}")
    print(f"Components: {', '.join(components)}")
    print(f"{'='*70}")

    # Run each component
    if "single" in components:
        print("\n\n=== COMPONENT 1: SINGLE AGENT (no debate, no RLM) ===")
        results = run_single_agent(problems, client, args.model, args.temperature)
        output_path = os.path.join(output_dir, f"single_{args.n_problems}.jsonl")
        with open(output_path, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"\nResults saved to: {output_path}")
        
        accuracy = sum(r["correct"] for r in results) / len(results)
        print(f"Accuracy: {accuracy:.1%}")

    if "debate" in components:
        print("\n\n=== COMPONENT 2: DEBATE ONLY (no RLM) ===")
        results = run_debate_only(problems, api_key, args.model, n_agents=3, 
                                   n_rounds=2, temperature=args.temperature, 
                                   base_url=args.base_url)
        output_path = os.path.join(output_dir, f"debate_{args.n_problems}.jsonl")
        with open(output_path, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"\nResults saved to: {output_path}")
        
        accuracy = sum(r["correct"] for r in results) / len(results)
        print(f"Accuracy: {accuracy:.1%}")

    if "rlm" in components:
        print("\n\n=== COMPONENT 3: RLM ONLY (no debate) ===")
        results = run_rlm_only(problems, api_key, args.model, args.temperature)
        output_path = os.path.join(output_dir, f"rlm_{args.n_problems}.jsonl")
        with open(output_path, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"\nResults saved to: {output_path}")
        
        accuracy = sum(r["correct"] for r in results) / len(results)
        print(f"Accuracy: {accuracy:.1%}")

    if "arlm" in components:
        print("\n\n=== COMPONENT 4: ARLM (debate + RLM) ===")
        results = run_arlm(problems, api_key, args.model, strategy="summary",
                           n_agents=3, n_rounds=4, temperature=args.temperature,
                           base_url=args.base_url)
        output_path = os.path.join(output_dir, f"arlm_{args.n_problems}.jsonl")
        with open(output_path, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"\nResults saved to: {output_path}")
        
        accuracy = sum(r["correct"] for r in results) / len(results)
        print(f"Accuracy: {accuracy:.1%}")

    print(f"\n{'='*70}")
    print(f"ABLATION STUDY COMPLETE")
    print(f"All results saved to: {output_dir}")
    print(f"\nRun analysis:")
    print(f"  python scripts/analyze_results.py {output_dir}/*.jsonl")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
