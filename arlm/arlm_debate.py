"""
ARLM: Adversarial Recursive Language Model debate engine.

Wraps multi-agent debate (Du et al. 2023) with RLM (Zhang et al. 2025)
so debate context is stored in a REPL environment instead of the prompt.
This removes the context ceiling that limited Du et al. to 2 rounds.

Classes:
    StandardDebate  – faithful reimplementation of Du et al.'s algorithm
                      using the modern OpenAI SDK.
    ARLMDebate      – debate engine with three context strategies:
                      "full"    (same as StandardDebate, for comparison)
                      "summary" (bounded context via progressive summaries)
                      "rlm"    (full RLM REPL integration)

Helper:
    extract_answer  – pull a numeric answer from \\boxed{...} or fallback patterns.
"""

import json
import os
import re
import sys
import time
import traceback
from typing import List, Optional, Dict, Any

from openai import OpenAI


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_answer(text: str) -> Optional[str]:
    """
    Extract a numerical answer from text.

    Tries, in order:
        1. \\boxed{<answer>}  (LaTeX, as required by the prompt)
        2. Last number that looks like a final answer

    Returns the answer as a cleaned string (no $ or commas), or None.
    """
    # 1. Try \\boxed{...}
    pattern = r"\\boxed\{([^}]*)\}"
    matches = re.findall(pattern, text)
    if matches:
        # Take the last \\boxed match (agents sometimes revise)
        answer = matches[-1]
        answer = re.sub(r"[^0-9.\-]", "", answer)
        if answer:
            return answer

    # 2. Fallback: look for {<number>} (Du et al. eval style)
    pattern2 = r"\{([0-9.,$ ]*)\}"
    matches2 = re.findall(pattern2, text)
    for m in reversed(matches2):
        cleaned = re.sub(r"[^0-9.\-]", "", m)
        if cleaned:
            return cleaned

    # 3. Last resort: last standalone number in text
    numbers = re.findall(r"[\-]?\d[\d,]*\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "")

    return None


def extract_gsm8k_answer(answer_text: str) -> Optional[str]:
    """
    Extract the ground-truth answer from a GSM8K answer field.

    GSM8K stores the answer as multi-line reasoning ending with
    ``#### <number>``.
    """
    pattern = r"####\s*([\-\d,\.]+)"
    match = re.search(pattern, answer_text)
    if match:
        return match.group(1).replace(",", "").strip()
    return None


def answers_match(pred: Optional[str], gold: Optional[str]) -> bool:
    """Compare two numeric answer strings with float tolerance."""
    if pred is None or gold is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-5
    except (ValueError, TypeError):
        return pred.strip() == gold.strip()


# ---------------------------------------------------------------------------
# API call helpers (retry with exponential backoff)
# ---------------------------------------------------------------------------

def _call_llm(client: OpenAI, model: str, messages: List[dict],
              temperature: float = 0.7, max_retries: int = 5) -> dict:
    """
    Call the chat completions API with retry logic.

    Returns a dict with keys:
        content          – the assistant's text
        prompt_tokens    – from usage (may be 0 for some providers)
        completion_tokens
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                n=1,
            )
            usage = response.usage
            return {
                "content": response.choices[0].message.content or "",
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
            }
        except Exception as e:
            last_error = e
            wait = min(2 ** attempt * 2, 60)
            print(f"  [retry {attempt+1}/{max_retries}] {type(e).__name__}: {e}  — waiting {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"API call failed after {max_retries} retries: {last_error}")


# ---------------------------------------------------------------------------
# StandardDebate — faithful reimplementation of Du et al.
# ---------------------------------------------------------------------------

class StandardDebate:
    """
    Reimplementation of Du et al.'s multi-agent debate.
    Uses the modern ``openai`` SDK (>=1.0).  Same algorithm and prompt
    templates as ``debate_src/gsm/gen_gsm.py``.
    """

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        n_agents: int = 3,
        n_rounds: int = 2,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
    ):
        self.model = model
        self.n_agents = n_agents
        self.n_rounds = n_rounds
        self.temperature = temperature
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url,
        )

    # -- prompt builders (matching Du et al. exactly) -----------------------

    @staticmethod
    def _initial_prompt(question: str) -> dict:
        """Round 0 user message — each agent sees only the question."""
        return {
            "role": "user",
            "content": (
                f"Can you solve the following math problem? {question} "
                "Explain your reasoning. Your final answer should be a "
                "single numerical number, in the form \\boxed{{answer}}, "
                "at the end of your response. "
            ),
        }

    @staticmethod
    def _debate_prompt(other_agents_responses: List[str], question: str) -> dict:
        """Round ≥1 user message — agent sees other agents' solutions."""
        prefix = "These are the solutions to the problem from other agents: "
        for resp in other_agents_responses:
            prefix += f"\n\n One agent solution: ```{resp}```"
        prefix += (
            "\n\n Using the solutions from other agents as additional "
            "information, can you provide your answer to the math problem? "
            f"\n The original math problem is {question}. Your final answer "
            "should be a single numerical number, in the form "
            "\\boxed{{answer}}, at the end of your response."
        )
        return {"role": "user", "content": prefix}

    @staticmethod
    def _final_check_prompt() -> dict:
        """Appended when ``len(agents) == 0`` in original code (unused in
        standard 3-agent runs, but kept for completeness)."""
        return {
            "role": "user",
            "content": (
                "Can you double check that your answer is correct. "
                "Please reiterate your answer, with your final answer a "
                "single numerical number, in the form \\boxed{{answer}}."
            ),
        }

    # -- main loop ----------------------------------------------------------

    def run(self, question: str) -> dict:
        """
        Run a full debate on *question*.

        Returns::

            {
                "final_answers": [str, ...],      # last response per agent
                "all_rounds":    [[str, ...], ...],# [round][agent]
                "agent_contexts": [...],           # full message lists
                "token_count":   int,              # sum of all tokens
                "wall_time":     float,            # seconds
            }
        """
        t0 = time.time()
        total_tokens = 0

        # Each agent maintains its own message list (exactly like Du et al.)
        agent_contexts: List[List[dict]] = [
            [self._initial_prompt(question)]
            for _ in range(self.n_agents)
        ]

        all_rounds: List[List[str]] = []  # [round_idx][agent_idx] = response text

        for rnd in range(self.n_rounds):
            round_responses: List[str] = []

            for i in range(self.n_agents):
                ctx = agent_contexts[i]

                # After round 0, inject other agents' most recent answers
                if rnd > 0:
                    prev_responses = []
                    for j in range(self.n_agents):
                        if j != i:
                            # The response from the previous round is the
                            # last assistant message in that agent's context.
                            # Index: 2*rnd - 1  matches the original code.
                            prev_responses.append(
                                agent_contexts[j][2 * rnd - 1]["content"]
                            )
                    msg = self._debate_prompt(prev_responses, question)
                    ctx.append(msg)

                result = _call_llm(
                    self.client, self.model, ctx,
                    temperature=self.temperature,
                )
                total_tokens += result["prompt_tokens"] + result["completion_tokens"]

                assistant_msg = {"role": "assistant", "content": result["content"]}
                ctx.append(assistant_msg)
                round_responses.append(result["content"])

            all_rounds.append(round_responses)

        wall_time = time.time() - t0

        return {
            "final_answers": [ctx[-1]["content"] for ctx in agent_contexts],
            "all_rounds": all_rounds,
            "agent_contexts": agent_contexts,
            "token_count": total_tokens,
            "wall_time": wall_time,
        }


# ---------------------------------------------------------------------------
# ARLMDebate — debate with RLM context management
# ---------------------------------------------------------------------------

class ARLMDebate:
    """
    Multi-agent debate with bounded context via RLM-style management.

    Strategies
    ----------
    ``"full"``    Identical to ``StandardDebate`` (baseline/control).
    ``"summary"`` After each round, generate a short summary of each agent's
                  position.  Next round sees: question + all prior summaries
                  + full text of only the most recent round.  Context per
                  call is bounded regardless of round count.
    ``"rlm"``    Full RLM REPL: the entire debate transcript is stored as a
                  context string in the REPL and each agent's response is
                  produced via ``RLM_REPL.completion()``.
    """

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        n_agents: int = 3,
        n_rounds: int = 2,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        context_strategy: str = "summary",
        temperature: float = 0.7,
    ):
        if context_strategy not in ("full", "summary", "rlm"):
            raise ValueError(f"Unknown context_strategy: {context_strategy!r}")

        self.model = model
        self.n_agents = n_agents
        self.n_rounds = n_rounds
        self.context_strategy = context_strategy
        self.temperature = temperature
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    # -- strategy: full (delegate to StandardDebate) ------------------------

    def _run_full(self, question: str) -> dict:
        sd = StandardDebate(
            model=self.model,
            n_agents=self.n_agents,
            n_rounds=self.n_rounds,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=self.temperature,
        )
        return sd.run(question)

    # -- strategy: summary --------------------------------------------------

    def _summarize_response(self, agent_idx: int, response: str) -> str:
        """
        Ask the LLM for a 1-2 sentence summary of an agent's response.
        Used to compress older rounds.
        """
        # Use a single user message (no system role) for broad model compat.
        msgs = [
            {
                "role": "user",
                "content": (
                    "You are a concise summariser. Given the following math "
                    "solution, produce a 1-2 sentence summary stating the "
                    "approach and the final numerical answer. Be brief.\n\n"
                    f"Solution:\n{response}"
                ),
            },
        ]
        result = _call_llm(self.client, self.model, msgs, temperature=0.3)
        return result["content"]

    def _build_summary_prompt(
        self,
        question: str,
        round_summaries: List[List[str]],
        last_round_responses: List[str],
        agent_idx: int,
        is_first_round: bool,
    ) -> List[dict]:
        """
        Build the message list for an agent under the *summary* strategy.

        Layout:
            [system] role description
            [user]   question + compressed history + recent full responses
        """
        if is_first_round:
            return [StandardDebate._initial_prompt(question)]

        # Build compressed history from older rounds
        history_parts: List[str] = []
        for rnd_idx, summaries in enumerate(round_summaries):
            for a_idx, summary in enumerate(summaries):
                history_parts.append(
                    f"Round {rnd_idx + 1}, Agent {a_idx + 1}: {summary}"
                )

        # Full text from the most recent round (other agents only)
        recent_parts: List[str] = []
        for a_idx, resp in enumerate(last_round_responses):
            if a_idx != agent_idx:
                recent_parts.append(f"Agent {a_idx + 1}: ```{resp}```")

        content = (
            f"Original math problem: {question}\n\n"
        )

        if history_parts:
            content += (
                "=== Summaries of earlier debate rounds ===\n"
                + "\n".join(history_parts)
                + "\n\n"
            )

        content += (
            "=== Most recent round (full responses from other agents) ===\n"
            + "\n\n".join(recent_parts)
            + "\n\n"
            "Using the above information, provide your answer to the math "
            "problem. Your final answer should be a single numerical number, "
            "in the form \\boxed{{answer}}, at the end of your response."
        )

        return [{"role": "user", "content": content}]

    def _run_summary(self, question: str) -> dict:
        t0 = time.time()
        total_tokens = 0

        all_rounds: List[List[str]] = []       # [round][agent] = full text
        round_summaries: List[List[str]] = []   # [round][agent] = summary

        for rnd in range(self.n_rounds):
            round_responses: List[str] = []

            for i in range(self.n_agents):
                is_first = (rnd == 0)
                # For the summary context, "last round" is the previous round
                last_round = all_rounds[-1] if all_rounds else []
                # Summaries of all rounds *before* the last full round
                older_summaries = round_summaries[:-1] if len(round_summaries) > 0 else []

                messages = self._build_summary_prompt(
                    question,
                    older_summaries,
                    last_round,
                    agent_idx=i,
                    is_first_round=is_first,
                )

                result = _call_llm(
                    self.client, self.model, messages,
                    temperature=self.temperature,
                )
                total_tokens += result["prompt_tokens"] + result["completion_tokens"]
                round_responses.append(result["content"])

            all_rounds.append(round_responses)

            # Summarize this round for future compression
            summaries: List[str] = []
            for a_idx, resp in enumerate(round_responses):
                try:
                    summary = self._summarize_response(a_idx, resp)
                    total_tokens += 100  # rough estimate for summary calls
                except Exception:
                    summary = resp[:200] + "..."
                summaries.append(summary)
            round_summaries.append(summaries)

        wall_time = time.time() - t0

        return {
            "final_answers": all_rounds[-1] if all_rounds else [],
            "all_rounds": all_rounds,
            "round_summaries": round_summaries,
            "token_count": total_tokens,
            "wall_time": wall_time,
            "context_strategy": "summary",
        }

    # -- strategy: rlm (full REPL) -----------------------------------------

    def _run_rlm(self, question: str) -> dict:
        """
        Use a single persistent RLM REPL to manage the entire debate.

        Architecture:
        - ONE ``REPLEnv`` is created at the start and persists across all
          rounds and all agent turns.
        - The debate transcript lives as a growing ``debate_transcript``
          variable inside the REPL.  Agents can programmatically read,
          grep, slice, and query it via generated Python code.
        - Each agent turn reuses the same REPL but gets a fresh LLM
          message history (so the model doesn't confuse its own prior
          turn with another agent's).  The REPL state (variables,
          transcript) carries forward.
        - After each agent responds, we append its response to the
          transcript variable *inside* the REPL via ``code_execution()``,
          so the next agent sees it immediately.

        This mirrors the original RLM design (one persistent REPL per
        completion) but extends it to multi-agent debate: the REPL is
        the shared workspace, agents are successive callers into it.
        """
        # Import RLM components from the vendored source
        rlm_src_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "rlm_src",
        )
        if rlm_src_path not in sys.path:
            sys.path.insert(0, rlm_src_path)

        from rlm.repl import REPLEnv  # noqa: E402
        from rlm.utils.llm import OpenAIClient  # noqa: E402
        from rlm.utils.prompts import next_action_prompt, build_system_prompt  # noqa: E402
        import rlm.utils.utils as rlm_utils  # noqa: E402

        t0 = time.time()
        total_tokens = 0
        all_rounds: List[List[str]] = []

        # --- Create ONE persistent REPL for the entire debate ---
        repl = REPLEnv(
            context_json=None,
            context_str=f"Math problem: {question}",
            recursive_model=self.model,
        )

        # Initialize the debate transcript as a REPL variable
        repl.code_execution('debate_transcript = context')  # starts with just the problem
        repl.code_execution('debate_round = 0')
        repl.code_execution(f'n_agents = {self.n_agents}')
        repl.code_execution(f'original_question = """{question}"""')

        # Create the LLM client (shared across agent turns)
        llm = OpenAIClient(
            api_key=self.api_key,
            model=self.model,
            base_url=self.base_url,
        )

        max_iterations_per_turn = 10

        for rnd in range(self.n_rounds):
            round_responses: List[str] = []

            # Update round counter in REPL
            repl.code_execution(f'debate_round = {rnd + 1}')

            for i in range(self.n_agents):
                # --- Fresh message history per agent turn ---
                # The REPL state persists, but each agent starts a new
                # conversation with the LLM so it reasons from scratch.
                messages = build_system_prompt()

                # Customize the system context for debate
                debate_context_msg = {
                    "role": "user",
                    "content": (
                        "You are participating in a multi-agent math debate.\n\n"
                        "The REPL environment contains:\n"
                        "- `original_question`: the math problem to solve\n"
                        "- `debate_transcript`: the full history of all agents' "
                        "responses across all rounds so far\n"
                        "- `debate_round`: the current round number "
                        f"({rnd + 1} of {self.n_rounds})\n"
                        f"- `n_agents`: the number of agents ({self.n_agents})\n"
                        f"\nYou are Agent {i + 1} of {self.n_agents} in "
                        f"Round {rnd + 1} of {self.n_rounds}.\n\n"
                        + (
                            "This is Round 1 — no prior debate history exists. "
                            "Read the original question from the REPL and solve it. "
                            if rnd == 0 else
                            "Read the debate_transcript from the REPL to see what "
                            "other agents have proposed. Consider their reasoning, "
                            "identify errors, and refine your answer. "
                        )
                        + "\nYour final answer MUST be a single numerical number "
                        "in the form \\boxed{answer}."
                    ),
                }
                messages.append(debate_context_msg)

                # --- Iterative REPL interaction (same as standard RLM) ---
                response_text = None
                for iteration in range(max_iterations_per_turn):
                    query_str = (
                        f"Solve the math problem as Agent {i+1}. "
                        "Use the REPL to read the question and any prior debate history."
                    )
                    prompt = next_action_prompt(query_str, iteration)
                    full_response = llm.completion(messages + [prompt])

                    # Check for code blocks to execute in the REPL
                    code_blocks = rlm_utils.find_code_blocks(full_response)

                    if code_blocks is not None:
                        messages = rlm_utils.process_code_execution(
                            full_response, messages, repl,
                            None, None,  # loggers (disabled)
                        )
                    else:
                        messages.append({
                            "role": "assistant",
                            "content": "You responded with:\n" + full_response,
                        })

                    # Check for final answer
                    final_answer = rlm_utils.check_for_final_answer(
                        full_response, repl, None,
                    )
                    if final_answer:
                        response_text = final_answer
                        break

                # If no final answer after max iterations, force one
                if response_text is None:
                    messages.append(next_action_prompt("", 0, final_answer=True))
                    response_text = llm.completion(messages)

                round_responses.append(response_text)

                # --- Append this agent's response to the REPL transcript ---
                # Escape the response for safe Python string insertion
                escaped = response_text.replace('\\', '\\\\').replace('"""', '\\"\\"\\"')
                repl.code_execution(
                    f'debate_transcript += "\\n\\n--- Round {rnd+1}, Agent {i+1} ---\\n" '
                    f'+ """{escaped}"""'
                )

            all_rounds.append(round_responses)

            # Add round separator to transcript
            repl.code_execution(
                f'debate_transcript += "\\n\\n=== End of Round {rnd+1} ===\\n"'
            )

        wall_time = time.time() - t0

        return {
            "final_answers": all_rounds[-1] if all_rounds else [],
            "all_rounds": all_rounds,
            "token_count": total_tokens,
            "wall_time": wall_time,
            "context_strategy": "rlm",
        }

    # -- public interface ---------------------------------------------------

    def run(self, question: str) -> dict:
        """
        Run debate with the configured context strategy.
        Same return schema as ``StandardDebate.run()``.
        """
        if self.context_strategy == "full":
            return self._run_full(question)
        elif self.context_strategy == "summary":
            return self._run_summary(question)
        elif self.context_strategy == "rlm":
            return self._run_rlm(question)
        else:
            raise ValueError(f"Unknown strategy: {self.context_strategy!r}")
