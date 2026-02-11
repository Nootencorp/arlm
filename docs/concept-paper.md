# Adversarial Recursive Language Models: A Novel Framework for Multi-Agent Debate via Programmatic Recursion

**Author:** Jeremy Nootenboom  
**Date:** February 2026  
**Status:** Concept Paper — v1 Draft

---

## Abstract

Large language models have achieved remarkable capabilities in text generation and reasoning, yet they remain brittle on complex, multi-step tasks requiring sustained logical coherence over long contexts. Two promising but disconnected research directions have emerged to address these limitations: **Recursive Language Models (RLMs)**, which embed LLMs in REPL environments to enable programmatic recursion and effectively unlimited context processing; and **adversarial multi-agent systems**, which use structured opposition — debate, critique, and repair loops — to systematically improve output quality. We propose **Adversarial Recursive Language Models (ARLM)**, a framework that fuses these paradigms by orchestrating adversarial agent roles (generator, auditor, planner, and self-repair) through RLM-style programmatic recursion with persistent shared REPL state. We argue this combination is more than the sum of its parts: persistent state enables richer adversarial interactions than verbal chains, recursive decomposition sidesteps context rot during extended debates, and heterogeneous model mixing across roles exploits architectural blind-spot diversity. As of February 2026, this specific combination appears unexplored in the published literature. We present the theoretical foundations, a detailed architectural design, four testable hypotheses, and a phased implementation roadmap. No experimental results are reported; this paper establishes the conceptual framework and invites empirical investigation.

---

## 1. Introduction

Large language models (LLMs) are now central to software engineering, research synthesis, content generation, and agentic workflows. Despite dramatic advances in scale and training methodology, a core fragility persists: performance degrades predictably on tasks that require sustained multi-step reasoning, self-correction over long contexts, or integration of contradictory evidence. A single model call, however sophisticated, remains fundamentally limited by the bounded context window and the absence of structured verification.

Two distinct research communities have attacked this problem from different angles:

**Recursive Language Models** represent a paradigm shift in how LLMs process long inputs. Introduced by Zhang, Kraska, and Khattab (2025), RLMs embed a language model within a Python REPL environment, storing the input context as a variable rather than stuffing it into the prompt. The model can then programmatically inspect, slice, search, and recursively sub-query itself over that context. This elegantly sidesteps context rot — the well-documented phenomenon where model quality degrades as token count increases — by ensuring no single model call ever needs to process the full context. Results are striking: RLM-Qwen3-8B, post-trained with only 1,000 training trajectories, outperforms the base Qwen3-8B by 28.3% on average and approaches vanilla GPT-5 quality on long-context benchmarks. An RLM using GPT-5-mini outperforms GPT-5 on the challenging OOLONG benchmark while being cheaper per query (Zhang et al., 2025).

**Adversarial multi-agent systems** take a fundamentally different approach: instead of improving how a single model processes input, they improve output quality through structured opposition. Du et al. (2023) demonstrated that multiple LLM instances proposing, debating, and refining responses over multiple rounds significantly enhances mathematical reasoning and reduces hallucinations. Liang et al. (2023) formalized Multi-Agent Debate (MAD), showing it overcomes the "Degeneration-of-Thought" problem where self-reflection with a single model fails to generate novel perspectives after initial confidence is established. This finding resonates with Huang et al.'s (2024) observation that LLMs struggle to self-correct reasoning without external feedback — a limitation that multi-agent opposition directly addresses.

**The gap is clear:** these two tracks have not been combined. RLM research to date has no adversarial or multi-agent component; each recursive call serves the same unified objective without structured opposition. Conversely, adversarial multi-agent systems operate through purely verbal, autoregressive exchanges — agents pass text back and forth without persistent shared state, programmatic inspection capabilities, or recursive decomposition of the adversarial process itself.

**We propose that the combination is more than the sum of its parts.** When adversarial agents share a persistent REPL state, a critic can programmatically `grep` through a generator's prior outputs, slice specific claims for targeted attack, and maintain structured convergence scores as code variables rather than verbal assessments. When adversarial debate is orchestrated through recursive decomposition, complex disputes can be broken into sub-debates that resolve independently before their conclusions are composed — mirroring how human expert panels actually reach consensus on complex technical questions.

This paper introduces **Adversarial Recursive Language Models (ARLM)**, outlines the theoretical advantages, presents a detailed architectural design, and proposes testable hypotheses for empirical validation.

---

## 2. Background and Related Work

### 2.1 Recursive Language Models

The RLM paradigm, proposed by Zhang, Kraska, and Khattab (2025), reconceptualizes the relationship between an LLM and its input context. Rather than treating the model as a black-box function $M(q, C)$ that takes a query and context and returns output, RLMs define a thin scaffold $RLM_M(q, C)$ where:

1. **Context is externalized.** The potentially massive context $C$ is stored as a Python variable in a REPL environment, not packed into the prompt.
2. **Programmatic access replaces token-stuffing.** The model can peek at subsets of $C$, run regex searches, compute statistics, and transform data — all through generated code executed in the REPL.
3. **Recursive self-calls enable decomposition.** The root model can spawn sub-RLM calls over transformed slices of context, each operating in an isolated REPL environment. Results flow back to the caller.

The key architectural insight is that the REPL environment is the medium through which the model interacts with context, and recursive LM calls are tools within that environment. The model decides its own chunking strategy, recursion pattern, and information aggregation approach at inference time — no rigid workflow is imposed.

Zhang et al. report that even with recursive depth limited to 1 (the root model can call sub-LMs but sub-LMs cannot call further sub-LMs), the approach handles inputs up to two orders of magnitude beyond model context windows. On the OOLONG benchmark with 132K+ tokens, RLM(GPT-5-mini) produces 33% more correct answers than GPT-5 at comparable cost. The post-trained RLM-Qwen3-8B demonstrates that the recursive reasoning trajectory is learnable through RL.

The official RLM library (`pip install rlms`) supports multiple LLM backends (OpenAI, Anthropic, open-source via vLLM), multiple sandbox environments (local, Docker, Modal, Prime Intellect), and provides trajectory visualization tools. A minimal implementation by the same authors and a community implementation by Gvadzabia (2025) further demonstrate the accessibility of the paradigm.

**Critically, all RLM work to date is single-agent.** The recursive calls serve one unified objective — there is no structured opposition, no adversarial pressure, and no mechanism for one model instance to challenge or audit another's reasoning.

### 2.2 Adversarial Multi-Agent Systems

The use of multiple LLM instances in structured opposition has emerged as a robust approach to improving reasoning quality.

**Multi-agent debate.** Du et al. (2023) introduced the foundational paradigm where multiple LLM instances propose individual responses, share them, debate over multiple rounds, and converge on a common answer. Their "society of minds" approach significantly improves mathematical and strategic reasoning while reducing hallucinations, and works directly with black-box models using identical procedures across tasks.

**Debate topology matters.** Liang et al. (2023) formalized this insight with their Multi-Agent Debate (MAD) framework, published at EMNLP 2024. They demonstrated that the *structure* of debate — including a judge role and "tit for tat" dynamics — matters more than raw agent count. They also identified that adaptive termination and moderate adversarial intensity are key to good performance. Importantly, they found that using different LLMs for different agent roles introduces fairness challenges, suggesting that role-architecture interaction is a meaningful design dimension.

**Scaling agent count.** Li et al. (2024) demonstrated with "Agent Forest" that LLM performance scales log-linearly with the number of agent instances via sampling-and-voting, published in TMLR. This scaling behavior is orthogonal to other enhancement methods and correlates with task difficulty — suggesting that adversarial approaches have the most headroom precisely on the hardest tasks.

**The self-correction problem.** Huang et al. (2024) demonstrated at ICLR 2024 that LLMs struggle to self-correct reasoning without external feedback — their performance sometimes *degrades* after self-correction attempts. This finding is foundational to the ARLM thesis: if single models cannot reliably self-correct, then external adversarial pressure from architecturally distinct models is not merely helpful but potentially *necessary* for reliable improvement.

**Heterogeneous model panels.** Verga et al. (2024) showed that a Panel of LLM evaluators (PoLL) composed of multiple smaller models from disjoint model families outperforms a single large judge model, exhibits less intra-model bias, and costs over seven times less. This directly supports the hypothesis that architectural diversity — different training data, different attention patterns, different failure modes — creates genuine blind-spot complementarity.

**LLM-as-Judge.** Zheng et al. (2023) established the LLM-as-a-Judge paradigm at NeurIPS 2023, demonstrating that strong LLM judges can match human preference agreement at 80%+. They also characterized key biases (position, verbosity, self-enhancement) that any adversarial system must account for.

**Self-rewarding systems.** Yuan et al. (2024) demonstrated Self-Rewarding Language Models at ICML 2024, where the language model provides its own training rewards via LLM-as-a-Judge prompting during iterative DPO training. This shows that adversarial self-evaluation can drive genuine capability improvement, though the approach is limited to single-model self-assessment.

**Limitations of current adversarial systems.** Despite these advances, existing multi-agent debate systems share critical limitations:

- **Context rot in extended debates.** As debate rounds accumulate, the growing transcript degrades model performance — the same fundamental problem RLMs were designed to solve.
- **Verbal scaffolding only.** Agents communicate through natural language, with no ability to programmatically inspect or verify prior claims. A critic must *re-read* a generator's output rather than `grep` through it.
- **No persistent shared state.** Each debate round starts from a growing text transcript rather than structured variables tracking proposals, scores, and convergence metrics.
- **Fixed communication topology.** Debate structure is predetermined, not adaptively determined by the models based on the problem's requirements.

### 2.3 Inference-Time Scaling

The ARLM framework draws significant motivation from the broader inference-time scaling paradigm. Snell et al. (2024) demonstrated that optimally scaling test-time compute can improve efficiency by 4× over best-of-N baselines, and in FLOPs-matched evaluations, a smaller model with sufficient test-time compute can outperform a 14× larger model. Brown et al. (2024) showed that coverage on coding tasks scales log-linearly with sample count over four orders of magnitude — DeepSeek-Coder-V2-Instruct goes from 15.9% to 56% on SWE-bench Lite with 250 samples versus one.

Li et al. (2024) provided theoretical grounding at ICLR 2024, proving that chain-of-thought reasoning empowers transformers to solve inherently serial problems that constant-depth transformers otherwise cannot — CoT with $T$ steps enables solving problems solvable by boolean circuits of size $T$.

These results collectively establish that inference-time compute is a legitimate scaling axis. ARLM extends this principle: recursive depth × adversarial debate rounds = multiplicatively more test-time compute, invested in structured verification rather than brute-force sampling.

### 2.4 Related Recursive Agent Systems

Several systems have explored recursive or hierarchical agent structures without combining programmatic recursion with adversarial dynamics:

**ReDel (Recursive Delegation).** ReDel-style approaches use recursive task delegation — a parent agent spawns child agents for sub-tasks — but without adversarial roles, REPL-based persistent state, or convergence through structured opposition. The delegation is cooperative, not adversarial.

**CodeAct and REPL-based agents.** CodeAct and similar systems embed LLMs in code execution environments, but without recursive self-calls or multi-agent adversarial dynamics. Zhang et al. (2025) note that RLMs build on CodeAct's design but fundamentally differ in treating context as an object to be understood through recursive decomposition.

**Self-RAG.** Asai et al. (2023) introduced Self-Reflective Retrieval-Augmented Generation, where a single model adaptively retrieves and self-reflects using special tokens. This is self-critique within a single model, not adversarial multi-agent opposition with persistent state.

**Self-Debugging.** Chen et al. (2023) showed that LLMs can debug their own code through rubber-duck debugging — explaining code to identify errors — achieving up to 12% improvements. This demonstrates the value of structured self-examination but remains single-model and does not leverage programmatic recursion or adversarial dynamics.

### 2.5 Gap in the Literature

To our knowledge, as of February 2026, **no published work combines REPL-based programmatic recursion with adversarial multi-agent convergence criteria.** Community extensions to RLM (e.g., Matryoshka-style nested models, async review pipelines) add multi-agent routing but not true adversarial loops. Broader agentic systems (2025–2026) use hierarchies, reflection, or tool-use, but not RLM-style programmatic recursion for adversarial dynamics.

This gap is not accidental — the two paradigms emerged from different communities (long-context processing vs. reasoning quality improvement) and address different failure modes (context rot vs. reasoning errors). ARLM proposes that these failure modes are deeply interrelated and that the solution to both is a unified framework.

---

## 3. Proposed Framework: Adversarial Recursive Language Models (ARLM)

### 3.1 Architecture Overview

ARLM orchestrates multiple agent roles, each instantiated as an RLM, operating over a shared persistent REPL state. The core roles are:

**Generator.** Produces initial proposals, solutions, or analyses. Operates as a full RLM with recursive decomposition capabilities — for complex tasks, it can spawn sub-generators for independent components.

**Auditor/Critic.** Adversarially examines generator outputs. Unlike verbal critics in existing debate systems, the ARLM auditor has *programmatic access* to the generator's outputs, intermediate states, and the original context. It can:
- `grep` through proposals for specific claims and cross-reference them against source material
- Execute code to verify quantitative assertions
- Slice specific sub-arguments for targeted recursive critique
- Maintain structured critique objects (not just text) with severity scores and confidence levels

**Planner.** Coordinates the decomposition strategy and debate topology. Rather than using a fixed debate structure, the planner adaptively determines:
- How to decompose the task for the generator
- Which aspects of the generator's output warrant adversarial scrutiny
- When to escalate sub-debates vs. accept convergence
- How to allocate compute budget across roles

**Self-Repair Agent.** Synthesizes generator output and auditor critiques into improved proposals. Operates with full visibility into the debate history (as structured data, not raw text) and can recursively invoke sub-repair for complex fixes.

### 3.2 Shared Persistent REPL State

The defining architectural innovation of ARLM is that all agents operate over a **shared REPL environment** where the state of the adversarial process is represented as structured data:

```python
# Shared ARLM State (illustrative)
proposals = []           # Generator outputs, versioned
critiques = []           # Auditor findings, structured
debate_history = []      # Full interaction log
convergence_scores = {}  # Per-claim agreement tracking
context = "..."          # Original input context (the RLM variable)
```

This is fundamentally different from verbal debate, where the entire history is a growing text transcript. In ARLM:

- **Proposals are versioned objects**, not text. The auditor can diff version 3 against version 2.
- **Critiques are structured**, with fields for claim-under-attack, evidence, severity, and suggested fix — not prose paragraphs.
- **Convergence is computed**, not assessed verbally. A claim converges when the auditor's critique severity drops below threshold *and* the generator's confidence exceeds threshold, tracked as numeric variables.

### 3.3 Recursive Opposition

ARLM enables **recursive adversarial dynamics** — a capability no existing system possesses:

1. **Sub-debates.** When the auditor identifies a contested claim, the system can spawn a focused sub-debate (sub-generator + sub-auditor) operating recursively over just that claim and its supporting evidence. This mirrors how human expert panels form sub-committees for contested technical points.

2. **Adversarial decomposition.** The planner can decompose not just the *task* but the *critique* — breaking a complex audit into independent sub-audits that each recursively verify a specific dimension (factual accuracy, logical consistency, completeness, etc.).

3. **Escalation.** If a sub-debate fails to converge, it can escalate to a higher-level debate round with access to broader context — the recursive structure provides natural escalation paths.

### 3.4 Comparison with Existing Approaches

| Dimension | Vanilla RLM | Multi-Agent Debate | ReDel-style Agents | Agent Forest | **ARLM (proposed)** |
|-----------|:-----------:|:-----------------:|:------------------:|:-----------:|:-------------------:|
| Adversarial roles | ❌ | ✅ | ❌ | ❌ | **✅** |
| Persistent REPL state | ✅ | ❌ | Partial | ❌ | **✅** |
| Recursive decomposition | ✅ | ❌ | ✅ | ❌ | **✅** |
| Scalable context | ✅ | ❌ | Partial | ❌ | **✅** |
| Programmatic inspection | ✅ | ❌ | Partial | ❌ | **✅** |
| Heterogeneous models | ❌ | Possible | Possible | ✅ | **✅ (by design)** |
| Adaptive topology | ❌ | ❌ | ❌ | ❌ | **✅** |
| Convergence criteria | N/A | Verbal | N/A | Voting | **Computed** |

### 3.5 Key Properties

**Emergent robustness through structured state.** When adversarial interactions are mediated through structured REPL variables rather than verbal chains, the quality of critique improves. A critic that can `re.findall(r'\\d+', proposal)` to extract all numeric claims and verify each against source data is more thorough than one that must parse prose to identify what to challenge.

**Context-rot resistance in extended debates.** This is perhaps ARLM's most distinctive advantage. In existing multi-agent debate systems, the transcript grows linearly with debate rounds, degrading model performance precisely when the debate gets interesting. In ARLM, each agent interacts with context through RLM-style programmatic access — the growing debate history is a structured variable to be queried, not a text blob to be re-read. Debate can continue for dozens of rounds without quality collapse.

**Inference-time scaling synergy.** ARLM provides two orthogonal axes of inference-time compute scaling: recursive depth (how deeply tasks and critiques decompose) and adversarial breadth (how many debate rounds occur). The product of these axes offers a richer compute-quality tradeoff surface than either alone.

**Heterogeneous model mixing.** ARLM is designed for different models per role. The generator might be a strong code-generation model (e.g., DeepSeek-Coder), the auditor a reasoning-focused model (e.g., Qwen3), and the planner a general-purpose model (e.g., Llama). Verga et al.'s (2024) finding that panels of diverse smaller models outperform single large judges directly motivates this design. Different architectures have genuinely different blind spots — a generator's systematic error patterns are less likely to be shared by an architecturally distinct auditor.

---

## 4. Implementation Roadmap

We propose a phased implementation that progressively introduces ARLM capabilities, with each phase independently valuable and empirically testable.

### Phase 1: REPL-Augmented Single Agent

**Goal:** Replace one-shot prompting with RLM-style REPL-based exploration for a single agent.

**Implementation:**
- Integrate the RLM library (`pip install rlms`) as the inference backbone
- Provide `read_file()`, `grep()`, `run_test()`, `apply_patch()` as REPL tools
- Agent writes code to explore context rather than receiving it in the prompt

**Success criteria:** Measurable improvement on long-context tasks versus vanilla prompting.

**Risk:** Low. This is essentially adopting the proven RLM approach as-is.

### Phase 2: Recursive Decomposition

**Goal:** Enable recursive self-calls for complex task decomposition.

**Implementation:**
- Complex tasks decompose into sub-tasks via recursive RLM calls
- Add a planner role (can be the same model with a planning system prompt) that determines decomposition strategy
- Track sub-task results as structured REPL variables

**Success criteria:** Improved performance on multi-step tasks that stall with flat approaches.

**Trigger for Phase 3:** System reliably completes basic cycles but produces errors that could be caught by structured review.

### Phase 3: Full Adversarial RLM Loop

**Goal:** Multiple adversarial roles debate through recursive calls until convergence.

**Implementation:**
- Instantiate generator, auditor, planner, and self-repair as separate RLM instances (potentially different models)
- Shared REPL state tracks proposals, critiques, and convergence scores
- Convergence criteria: all roles agree (auditor critique severity below threshold) OR maximum debate rounds reached
- Planner determines debate topology adaptively based on task complexity

**Success criteria:** Outperformance versus Phase 2 on tasks requiring self-correction, factual accuracy, or logical consistency.

---

## 5. Expected Advantages and Hypotheses

We propose four specific, testable hypotheses:

### H1: ARLM outperforms vanilla RLM on self-correction tasks

**Claim:** On tasks requiring multi-step reasoning with self-correction (e.g., multi-hop question answering with planted contradictions, code generation with adversarial test suites), ARLM will achieve higher accuracy than vanilla single-agent RLM.

**Rationale:** Huang et al. (2024) demonstrated that single models cannot reliably self-correct without external feedback. ARLM provides that external feedback through architecturally distinct auditor models. The RLM substrate ensures this feedback loop operates without context degradation.

**Proposed test:** Compare ARLM vs. vanilla RLM on a curated benchmark of multi-hop reasoning tasks where inputs contain deliberately planted contradictions or misleading information. Measure accuracy, error detection rate, and self-correction success rate.

### H2: ARLM outperforms non-recursive debate on long-context tasks

**Claim:** On tasks involving adversarial debate over long contexts (>100K tokens), ARLM will maintain higher quality across debate rounds than standard multi-agent debate systems that use verbal transcript accumulation.

**Rationale:** Standard debate systems accumulate a growing text transcript that triggers context rot — the same phenomenon that RLMs were designed to eliminate. ARLM's structured REPL state and programmatic context access should prevent quality degradation as debate length increases.

**Proposed test:** Compare ARLM vs. standard MAD on the OOLONG benchmark extended with adversarial debate requirements (e.g., "find and resolve all contradictions in this 200K-token corpus"). Measure quality as a function of debate round count, specifically tracking the inflection point where standard debate degrades.

### H3: Heterogeneous model mixing amplifies gains within ARLM

**Claim:** ARLM with architecturally diverse models across roles (e.g., generator from one model family, auditor from another) will outperform ARLM with homogeneous models by a margin exceeding what either diversity or ARLM structure contributes alone.

**Rationale:** Verga et al. (2024) demonstrated that panels of diverse smaller models outperform single large models due to reduced intra-model bias. Within ARLM, heterogeneous models should exhibit genuinely complementary failure modes — a generator's systematic blind spots are unlikely to be shared by an auditor from a different model family, making the adversarial pressure more effective.

**Proposed test:** Full factorial design: {ARLM, standard debate} × {homogeneous models, heterogeneous models} on reasoning benchmarks. Test for super-additive interaction between ARLM structure and model diversity.

### H4: ARLM with open models approaches frontier model performance

**Claim:** ARLM orchestrating free/open-weight models (e.g., Qwen3-8B, Llama-3.1-8B, Mistral-7B) can approach the performance of single frontier model calls (e.g., GPT-5, Claude) on complex reasoning tasks, through inference-time compute scaling.

**Rationale:** Snell et al. (2024) showed that test-time compute scaling can enable a smaller model to outperform a 14× larger one. Brown et al. (2024) demonstrated log-linear coverage scaling with sample count. ARLM provides a structured, non-random way to invest inference-time compute — adversarial verification should be more sample-efficient than brute-force repeated sampling.

**Proposed test:** Compare ARLM(Qwen3-8B + Llama-3.1-8B + Mistral-7B) vs. single-call GPT-5 on GSM8K-hard, MATH, and SWE-bench Lite. Control for total FLOPs consumed. Measure the compute-quality tradeoff curve.

---

## 6. Practical Considerations

### 6.1 Works with Free and Open Models

ARLM is explicitly designed for accessibility. The RLM library supports any OpenAI-compatible API, including local models via vLLM and Ollama. The adversarial multi-agent layer requires no model-specific capabilities beyond standard instruction following and code generation. A minimal ARLM deployment could use:

- **Generator:** Qwen3-8B (strong general reasoning, natively recursive after post-training)
- **Auditor:** Llama-3.1-8B (different architecture family, good at following structured critique instructions)
- **Planner:** Mistral-7B (efficient, good at structured output)

Total cost: free for locally hosted models, or minimal API cost for cloud-hosted open models. This is critical for the framework's adoption — the value proposition increases as the cost of frontier models decreases the budget available for multi-call approaches.

### 6.2 Heterogeneous Model Selection Strategy

Not all model combinations will be equally effective. We propose the following selection principles:

1. **Maximize architectural diversity.** Choose models from different families (e.g., Qwen, Llama, Mistral) to maximize blind-spot complementarity. Models trained on different data with different architectures and different RLHF procedures will have genuinely different failure modes.

2. **Match capability to role.** The generator should be strong at the target task domain. The auditor should be strong at structured analysis and critique. The planner needs good instruction following and structured output.

3. **Consider the cost-quality surface.** Use larger models for roles that bottleneck quality (often the auditor, whose missed errors propagate) and smaller models for roles where throughput matters (often the planner, which makes routing decisions).

4. **Empirically calibrate.** Run a small calibration set to identify which model combinations produce the most productive adversarial tension — not too agreeable (rubber-stamping), not too adversarial (endless unresolvable disagreement).

### 6.3 Convergence Criteria Design

The convergence mechanism is central to ARLM's practical viability. We propose a multi-signal convergence function:

1. **Critique severity threshold.** Convergence when the auditor's maximum critique severity score drops below a task-specific threshold. Severity is a structured numeric field, not a verbal assessment.

2. **Generator confidence threshold.** The generator assigns confidence scores to its claims. Convergence requires confidence above threshold on all claims the auditor has reviewed.

3. **Delta threshold.** Convergence when successive proposal versions show diminishing changes (measured by structured diff, not text similarity).

4. **Maximum rounds.** Hard cap to prevent runaway debate. This should be calibrated per task complexity — simple tasks need 2-3 rounds, complex tasks may need 10+.

5. **Escalation trigger.** If a sub-debate fails to converge within its round budget, it escalates to a higher level rather than being forcibly terminated.

The convergence function is itself a designable component of the system — future work could train a convergence predictor that learns to estimate when additional debate rounds will yield diminishing returns.

### 6.4 Compute Budget Management

ARLM trades additional inference cost for quality improvement. Practical deployment requires:

- **Adaptive compute allocation.** Easy tasks should converge in 1-2 rounds with minimal recursive depth. Hard tasks should automatically receive more compute. The planner role explicitly manages this tradeoff.
- **Early termination.** If the auditor finds no significant issues in the first pass, skip further debate rounds.
- **Cost monitoring.** Track per-task cost (API calls, tokens consumed) and expose this as a tunable parameter.

---

## 7. Evaluation Plan

### 7.1 Baselines

We propose evaluation against the following baselines:

1. **Vanilla single-model call** (e.g., GPT-5, Claude)
2. **Vanilla RLM** — single-agent recursive processing (Zhang et al., 2025)
3. **Standard multi-agent debate** — Du et al. (2023) and Liang et al. (2023) frameworks
4. **Agent Forest** — sampling-and-voting scaling (Li et al., 2024)
5. **Best-of-N** — repeated sampling with selection (Brown et al., 2024)

### 7.2 Target Tasks

- **Long-document consistency checking:** Given a 100K+ token corpus with planted contradictions, identify and resolve all inconsistencies.
- **Multi-hop research synthesis with adversarial distractors:** Answer complex questions requiring integration of evidence across multiple documents, with deliberately misleading passages included.
- **Complex code generation with adversarial test suites:** Generate code that passes both standard and adversarial test cases designed to exploit common LLM coding errors.
- **Extended agentic tasks:** Multi-turn coding sessions (e.g., SWE-bench) where context rot typically degrades performance.

### 7.3 Metrics

- **Accuracy / task success rate** (primary)
- **Convergence rate:** Fraction of tasks where adversarial debate reaches genuine convergence vs. hitting the round cap
- **Token efficiency:** Quality per total token consumed, compared across methods
- **Wall-clock time:** Practical latency for interactive use cases
- **Quality vs. debate rounds:** The degradation curve (or improvement curve) as debate length increases
- **Error detection rate:** Fraction of planted errors / contradictions identified by the auditor

---

## 8. Novelty Assessment and Honest Limitations

### 8.1 What Is Novel

The specific combination of REPL-based programmatic recursion with adversarial multi-agent convergence criteria appears to be unexplored as of February 2026. The individual components — RLMs, multi-agent debate, heterogeneous model panels — are established. The claimed novelty is in their integration and the emergent properties that integration enables (programmatic adversarial inspection, context-rot-resistant debate, recursive sub-debates).

### 8.2 What Could Challenge Novelty

- **Concurrent work.** The pace of AI research in 2025-2026 makes it possible that similar combinations are being developed independently. We have not found published evidence of this, but the search was limited by web search API availability.
- **Industrial systems.** Proprietary agentic systems (e.g., within Anthropic, OpenAI, Google) may already implement similar patterns internally without publication.
- **Incremental vs. fundamental.** A skeptic could argue that ARLM is "just" running RLMs in a multi-agent loop — an engineering combination rather than a conceptual advance. We counter that the emergent properties (programmatic adversarial inspection, structured convergence, recursive sub-debates) are qualitatively new capabilities that neither paradigm provides alone.

### 8.3 Limitations

- **No experimental results.** This is a concept paper. All claims about performance advantages are theoretical and require empirical validation.
- **Compute cost.** ARLM will consume more inference compute than single-model approaches. The hypothesis is that the quality improvement justifies the cost, but this must be demonstrated.
- **Convergence is not guaranteed.** Adversarial debate can fail to converge, produce oscillating disagreements, or converge on wrong answers if all models share a blind spot.
- **Prompt engineering complexity.** Defining effective system prompts for each role, calibrating convergence thresholds, and selecting model combinations introduces significant design complexity.

---

## 9. Conclusion

Recursive Language Models and adversarial multi-agent systems are two powerful paradigms that have, until now, developed in isolation. We have argued that their combination — Adversarial Recursive Language Models — is theoretically well-motivated, architecturally feasible with today's tools and open-weight models, and likely to produce capabilities that neither paradigm provides alone.

The framework addresses real limitations in both parent paradigms: RLMs gain structured adversarial pressure that compensates for single-model blind spots, while adversarial debate systems gain context-rot resistance and programmatic inspection capabilities that enable deeper, more rigorous opposition.

We have presented four testable hypotheses, a phased implementation roadmap, and practical guidance for model selection, convergence criteria, and compute management. The framework is implementable today using the open-source RLM library and freely available language models.

We invite the research community to empirically investigate this direction. The code, when developed, will be released openly.

---

## References

1. **Zhang, A. L., Kraska, T., & Khattab, O.** (2025). "Recursive Language Models." *arXiv preprint* arXiv:2512.24601. [https://arxiv.org/abs/2512.24601](https://arxiv.org/abs/2512.24601)  
   — Introduces the RLM paradigm. RLM-Qwen3-8B outperforms base Qwen3-8B by 28.3%; RLM(GPT-5-mini) outperforms GPT-5 on OOLONG benchmark. Code: [https://github.com/alexzhang13/rlm](https://github.com/alexzhang13/rlm)

2. **Du, Y., Li, S., Torralba, A., Tenenbaum, J. B., & Mordatch, I.** (2023). "Improving Factuality and Reasoning in Language Models through Multiagent Debate." *arXiv preprint* arXiv:2305.14325. [https://arxiv.org/abs/2305.14325](https://arxiv.org/abs/2305.14325)  
   — Foundational multi-agent debate paper. Shows debate significantly improves mathematical reasoning and reduces hallucinations.

3. **Liang, T., He, Z., Jiao, W., Wang, X., Wang, Y., Wang, R., Yang, Y., Tu, Z., & Shi, S.** (2023). "Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate." *EMNLP 2024*. arXiv:2305.19118. [https://arxiv.org/abs/2305.19118](https://arxiv.org/abs/2305.19118)  
   — Formalizes MAD framework. Demonstrates that debate topology and adaptive termination are critical; identifies Degeneration-of-Thought problem in self-reflection.

4. **Huang, J., Gu, S. S., Le, H., Hasan, M., Shin, S., & Pfister, T.** (2024). "Large Language Models Cannot Self-Correct Reasoning Yet." *ICLR 2024*. arXiv:2310.01798. [https://arxiv.org/abs/2310.01798](https://arxiv.org/abs/2310.01798)  
   — Shows LLMs struggle to self-correct without external feedback; performance sometimes degrades after self-correction attempts.

5. **Snell, C., Lee, J., Xu, K., & Kumar, A.** (2024). "Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters." *arXiv preprint* arXiv:2408.03314. [https://arxiv.org/abs/2408.03314](https://arxiv.org/abs/2408.03314)  
   — Demonstrates compute-optimal test-time scaling: 4× efficiency improvement over best-of-N; smaller models with test-time compute can outperform 14× larger models.

6. **Brown, B., Juravsky, J., Ehrlich, R., Clark, R., Le, Q. V., Ré, C., & Mirhoseini, A.** (2024). "Large Language Monkeys: Scaling Inference Compute with Repeated Sampling." *arXiv preprint* arXiv:2407.21787. [https://arxiv.org/abs/2407.21787](https://arxiv.org/abs/2407.21787)  
   — Coverage scales log-linearly with samples over four orders of magnitude. DeepSeek-Coder-V2-Instruct: 15.9% → 56% on SWE-bench Lite with 250 samples.

7. **Verga, P., Hofstätter, S., Althammer, S., Wan, Y., Sil, A., & Allan, J.** (2024). "Replacing Judges with Juries: Evaluating LLM Generations with a Panel of Diverse Models." *arXiv preprint* arXiv:2404.18796. [https://arxiv.org/abs/2404.18796](https://arxiv.org/abs/2404.18796)  
   — Panel of LLM evaluators (PoLL) from disjoint model families outperforms single large judge; 7× cheaper with less intra-model bias.

8. **Zheng, L., Chiang, W.-L., Sheng, Y., et al.** (2023). "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena." *NeurIPS 2023 Datasets and Benchmarks*. arXiv:2306.05685. [https://arxiv.org/abs/2306.05685](https://arxiv.org/abs/2306.05685)  
   — Establishes LLM-as-a-Judge paradigm. GPT-4 matches human preference agreement at 80%+. Identifies position, verbosity, and self-enhancement biases.

9. **Yuan, W., Pang, R. Y., Cho, K., Sukhbaatar, S., Xu, J., & Weston, J.** (2024). "Self-Rewarding Language Models." *ICML 2024*. arXiv:2401.10020. [https://arxiv.org/abs/2401.10020](https://arxiv.org/abs/2401.10020)  
   — LLM provides its own rewards via LLM-as-a-Judge during iterative DPO training. Shows both instruction following and reward quality improve simultaneously.

10. **Li, J., Zhang, B., Guo, Y., Liu, J., Ding, J., & Ye, D.** (2024). "More Agents Is All You Need." *Transactions on Machine Learning Research (TMLR)*. arXiv:2402.05120. [https://arxiv.org/abs/2402.05120](https://arxiv.org/abs/2402.05120)  
    — Performance scales with agent count via sampling-and-voting (Agent Forest). Scaling correlates with task difficulty.

11. **Li, Z., Cai, B., Chen, Y., & Liang, Y.** (2024). "Chain of Thought Empowers Transformers to Solve Inherently Serial Problems." *ICLR 2024*. arXiv:2402.12875. [https://arxiv.org/abs/2402.12875](https://arxiv.org/abs/2402.12875)  
    — Theoretical proof that CoT with $T$ steps enables solving problems solvable by boolean circuits of size $T$.

12. **Chen, X., Lin, M., Schärli, N., & Zhou, D.** (2023). "Teaching Large Language Models to Self-Debug." *arXiv preprint* arXiv:2304.05128. [https://arxiv.org/abs/2304.05128](https://arxiv.org/abs/2304.05128)  
    — Self-Debugging via rubber duck debugging. Up to 12% accuracy improvement on code translation tasks.

13. **Asai, A., Wu, Z., Wang, Y., Sil, A., & Hajishirzi, H.** (2023). "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection." *arXiv preprint* arXiv:2310.11511. [https://arxiv.org/abs/2310.11511](https://arxiv.org/abs/2310.11511)  
    — Self-Reflective RAG using special reflection tokens for adaptive retrieval and self-critique.

14. **Chen, H., Feng, Y., Liu, Z., et al.** (2024). "Language Models are Hidden Reasoners: Unlocking Latent Reasoning Capabilities via Self-Rewarding." *arXiv preprint* arXiv:2411.04282. [https://arxiv.org/abs/2411.04282](https://arxiv.org/abs/2411.04282)  
    — LaTRO: Latent Reasoning Optimization improves zero-shot accuracy by 12.5% on GSM8K through self-rewarding without external feedback.

15. **Zhang, A. L.** (2025). "Recursive Language Models." *Blog post*. [https://alexzhang13.github.io/blog/2025/rlm](https://alexzhang13.github.io/blog/2025/rlm)  
    — Original blog introducing RLM concept with early experimental results.

16. **Gvadzabia, G.** (2025). "recursive-llm: Python Implementation of Recursive Language Models." *GitHub*. [https://github.com/ysz/recursive-llm](https://github.com/ysz/recursive-llm)  
    — Community RLM implementation using LiteLLM for universal model support. Demonstrates 80% accuracy vs. 0% for direct OpenAI on 60K-token structured queries.

---

*This paper proposes a conceptual framework. No experimental results are reported. All hypotheses require empirical validation. The author welcomes collaboration and critique.*
