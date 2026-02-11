#!/usr/bin/env python3
"""
Smoke test for ARLM basic debate mechanics

Tests a simple 2-agent, 1-round debate on a GSM8K-style math problem
using OpenRouter free models.

SAFETY: Uses OPENROUTER_API_KEY from environment, never writes it to disk.
"""

import os
import sys
from openai import OpenAI

# Verify API key is available
if "OPENROUTER_API_KEY" not in os.environ:
    print("ERROR: OPENROUTER_API_KEY not found in environment")
    print("Run: source /home/nootencorp/dual-agent-loop/.env")
    sys.exit(1)

# Initialize OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"]
)

# Test problem: simple GSM8K-style math
PROBLEM = """
Sarah has 3 times as many apples as Tom. 
Tom has 12 apples. 
How many apples does Sarah have?
"""

def run_agent(agent_name, problem, previous_answer=None):
    """Run a single agent's reasoning."""
    
    if previous_answer:
        prompt = f"""You are Agent {agent_name} in a debate.

Problem: {problem}

Agent {previous_answer['agent']} said: {previous_answer['answer']}

Do you agree? Provide your own answer and reasoning. Be concise."""
    else:
        prompt = f"""You are Agent {agent_name}.

Problem: {problem}

Provide your answer and reasoning. Be concise."""
    
    # Try multiple free models in order of preference
    free_models = [
        "mistralai/mistral-7b-instruct:free",
        "meta-llama/llama-3-8b-instruct:free",
        "google/gemma-2-9b-it:free"
    ]
    
    for model in free_models:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            if model == free_models[-1]:  # Last model failed
                print(f"\n⚠️  All free models failed. Last error: {e}")
                print("Note: Free models may be temporarily rate-limited.")
                print("The smoke test structure is correct - API availability is the issue.")
                raise
            continue  # Try next model

def main():
    print("=" * 60)
    print("ARLM Smoke Test: 2-Agent Debate")
    print("=" * 60)
    print(f"\nProblem: {PROBLEM.strip()}")
    print("\n" + "-" * 60)
    
    # Round 1: Agent A proposes initial answer
    print("\n[Agent A - Initial Answer]")
    answer_a = run_agent("A", PROBLEM)
    print(answer_a)
    print("\n" + "-" * 60)
    
    # Round 1: Agent B responds
    print("\n[Agent B - Response]")
    answer_b = run_agent("B", PROBLEM, {"agent": "A", "answer": answer_a})
    print(answer_b)
    print("\n" + "-" * 60)
    
    print("\n✅ Smoke test completed successfully!")
    print("Basic debate mechanics are working.")
    print("\nNext steps:")
    print("  - Implement RLMS judge for preference signals")
    print("  - Add multi-round debate logic")
    print("  - Integrate with debate_src benchmarks")
    print("=" * 60)

if __name__ == "__main__":
    main()
