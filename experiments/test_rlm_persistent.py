#!/usr/bin/env python3
"""Quick validation: ARLM RLM strategy with persistent shared REPL.

Runs 2 GSM8K problems with gpt-4.1-nano, 2 agents, 2 rounds.
Goal: verify architecture works end-to-end, not accuracy.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv("/home/nootencorp/nootencorp-agent-worktrees/rsi-runner/.env")

from arlm.arlm_debate import ARLMDebate, extract_gsm8k_answer, extract_answer, answers_match

MODEL = "gpt-4.1-nano"
N_AGENTS = 2
N_ROUNDS = 2
N_PROBLEMS = 2

def load_gsm8k(n):
    with open("benchmarks/gsm8k/test.jsonl") as f:
        return [json.loads(line) for _, line in zip(range(n), f)]

def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    problems = load_gsm8k(N_PROBLEMS)
    
    print(f"=== ARLM RLM Persistent REPL Test ===")
    print(f"Model: {MODEL} | Agents: {N_AGENTS} | Rounds: {N_ROUNDS} | Problems: {N_PROBLEMS}")
    print()

    correct = 0
    for idx, prob in enumerate(problems):
        question = prob["question"]
        gold = extract_gsm8k_answer(prob["answer"])
        
        print(f"--- Problem {idx+1}/{N_PROBLEMS} ---")
        print(f"Q: {question[:80]}...")
        print(f"Gold: {gold}")
        
        debate = ARLMDebate(
            model=MODEL,
            api_key=api_key,
            n_agents=N_AGENTS,
            n_rounds=N_ROUNDS,
            context_strategy="rlm",
            early_exit=False,
        )
        
        t0 = time.time()
        try:
            result = debate.run(question)
            elapsed = time.time() - t0
            
            final_answers = result["final_answers"]
            print(f"Final answers: {final_answers}")
            print(f"Rounds used: {result['rounds_used']}")
            print(f"Time: {elapsed:.1f}s")
            
            # Majority vote - use extract_answer for model output, extract_gsm8k_answer is for gold only
            from collections import Counter
            extracted = [extract_answer(a) for a in final_answers]
            print(f"Extracted: {extracted}")
            
            if extracted:
                majority = Counter(extracted).most_common(1)[0][0]
                is_correct = answers_match(majority, gold)
                print(f"Majority: {majority} | Correct: {is_correct}")
                if is_correct:
                    correct += 1
            else:
                print("No answers extracted!")
                
        except Exception as e:
            elapsed = time.time() - t0
            print(f"ERROR after {elapsed:.1f}s: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
        
        print()
    
    print(f"=== Results: {correct}/{N_PROBLEMS} correct ===")

if __name__ == "__main__":
    main()
