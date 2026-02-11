#!/usr/bin/env python3
"""Debug version: trace each step of the RLM REPL loop."""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv("/home/nootencorp/nootencorp-agent-worktrees/rsi-runner/.env")

# Patch the _run_rlm method to add debug prints
from arlm.arlm_debate import ARLMDebate, extract_gsm8k_answer

MODEL = "gpt-4.1-nano"
N_AGENTS = 2
N_ROUNDS = 2

def load_gsm8k(n):
    with open("benchmarks/gsm8k/test.jsonl") as f:
        return [json.loads(line) for _, line in zip(range(n), f)]

def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    problems = load_gsm8k(1)
    prob = problems[0]
    question = prob["question"]
    gold = extract_gsm8k_answer(prob["answer"])

    print(f"Q: {question[:80]}...")
    print(f"Gold: {gold}")
    print()

    # Import RLM components directly to debug
    rlm_src_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "rlm_src",
    )
    sys.path.insert(0, rlm_src_path)

    from rlm.repl import REPLEnv
    from rlm.utils.llm import OpenAIClient
    from rlm.utils.prompts import next_action_prompt, build_system_prompt
    import rlm.utils.utils as rlm_utils

    # Create REPL
    print("[1] Creating REPL...")
    repl = REPLEnv(
        context_json=None,
        context_str=f"Math problem: {question}",
        recursive_model=MODEL,
    )
    print(f"[1] REPL created. Type: {type(repl)}")

    # Init vars
    print("[2] Initializing REPL variables...")
    repl.code_execution(f'original_question = """{question}"""')
    repl.code_execution('debate_transcript = ""')
    repl.code_execution('debate_round = 0')
    repl.code_execution(f'n_agents = {N_AGENTS}')
    repl.code_execution(
        'context = "ORIGINAL PROBLEM:\\n" + original_question '
        '+ "\\n\\nDEBATE HISTORY:\\n" + debate_transcript'
    )
    print("[2] Variables initialized")

    # Create LLM client
    print("[3] Creating LLM client...")
    llm = OpenAIClient(api_key=api_key, model=MODEL)
    print(f"[3] LLM client created: {type(llm)}")

    # Round 1, Agent 1
    print("\n=== Round 1, Agent 1 ===")
    repl.code_execution('debate_round = 1')

    messages = build_system_prompt()
    print(f"[4] System prompt has {len(messages)} messages")

    debate_context_msg = {
        "role": "user",
        "content": (
            "You are participating in a multi-agent math debate.\n\n"
            "The REPL environment contains:\n"
            "- `original_question`: the math problem to solve\n"
            "- `debate_transcript`: the full history of all agents' responses\n"
            "- `debate_round`: current round (1 of 2)\n"
            f"- `n_agents`: number of agents ({N_AGENTS})\n"
            f"\nYou are Agent 1 of {N_AGENTS} in Round 1 of {N_ROUNDS}.\n\n"
            "This is Round 1 — no prior debate history exists. "
            "Read the original question from the REPL and solve it. "
            "\nYour final answer MUST be a single numerical number "
            "in the form \\boxed{answer}."
        ),
    }
    messages.append(debate_context_msg)

    max_iters = 10
    response_text = None

    for iteration in range(max_iters):
        query_str = "Solve the math problem as Agent 1. Use the REPL to read the question and any prior debate history."
        prompt = next_action_prompt(query_str, iteration)
        print(f"\n[iter {iteration}] Calling LLM... (msgs={len(messages) + 1})")
        t0 = time.time()
        full_response = llm.completion(messages + [prompt])
        elapsed = time.time() - t0
        print(f"[iter {iteration}] LLM responded in {elapsed:.1f}s")
        print(f"[iter {iteration}] Response preview: {full_response[:200]}...")

        code_blocks = rlm_utils.find_code_blocks(full_response)
        print(f"[iter {iteration}] Code blocks found: {code_blocks is not None}")

        if code_blocks is not None:
            print(f"[iter {iteration}] Executing code in REPL...")
            messages = rlm_utils.process_code_execution(
                full_response, messages, repl, None, None,
            )
            print(f"[iter {iteration}] Code executed. msgs now: {len(messages)}")
        else:
            messages.append({
                "role": "assistant",
                "content": "You responded with:\n" + full_response,
            })

        final_answer = rlm_utils.check_for_final_answer(full_response, repl, None)
        print(f"[iter {iteration}] Final answer check: {final_answer}")
        if final_answer:
            response_text = final_answer
            break

    if response_text is None:
        print("\n[!] No final answer after max iterations, forcing...")
        messages.append(next_action_prompt("", 0, final_answer=True))
        response_text = llm.completion(messages)

    print(f"\n=== Agent 1 final response ===")
    print(response_text[:500])
    extracted = extract_gsm8k_answer(response_text)
    print(f"Extracted: {extracted}, Gold: {gold}, Match: {extracted == gold}")

if __name__ == "__main__":
    main()
