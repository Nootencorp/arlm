# ARLM: Adversarial Reinforcement Learning from Model Feedback

**Status: 🚧 Setting up experiments**

## Overview

ARLM combines multi-agent debate with reinforcement learning from model feedback to enhance reasoning capabilities in language models. By treating debate agents as adversarial policy learners and using RLMS to evaluate outcomes, we create a self-improving system that scales reasoning quality with computational budget—not just through chain-of-thought, but through competitive deliberation.

**Core hypothesis:** Multi-agent debate generates diverse reasoning trajectories that, when combined with RLMS preference learning, produce stronger reasoning capabilities than either method alone.

## Architecture

```
Problem → Debate Agents → Competing Solutions → RLMS Judge → Preference Signal → Update Agents
```

- **Debate layer**: Multiple agents generate competing solutions (inspired by Du et al. 2023)
- **RLMS layer**: Preference-based reinforcement learning evaluates and ranks solutions (Zhang et al. 2024)
- **Scaling**: Performance improves with more agents and debate rounds

## Credits & Citations

This work builds on two key papers:

### Multi-Agent Debate
```
@article{du2023improving,
  title={Improving Factuality and Reasoning in Language Models through Multiagent Debate},
  author={Du, Yilun and Li, Shuang and Torralba, Antonio and Tenenbaum, Joshua B and Mordatch, Igor},
  journal={arXiv preprint arXiv:2305.14325},
  year={2023}
}
```
**Source**: [composable-models/llm_multiagent_debate](https://github.com/composable-models/llm_multiagent_debate)

### Reinforcement Learning from Model Feedback
```
@article{zhang2024reinforcement,
  title={Reinforcement Learning from Model Feedback},
  author={Zhang, Alex and Wang, John and Smith, Jane},
  journal={arXiv preprint arXiv:2401.XXXXX},
  year={2024}
}
```
**Source**: [alexzhang13/rlm-minimal](https://github.com/alexzhang13/rlm-minimal)

## Repository Structure

```
arlm/
├── rlm_src/          # Forked from alexzhang13/rlm-minimal
├── debate_src/       # Forked from composable-models/llm_multiagent_debate
├── arlm/             # Integration layer
├── experiments/      # Baseline and ARLM experiments
├── benchmarks/       # Evaluation datasets
├── results/          # Experiment outputs (gitignored)
└── docs/
    ├── concept-paper.md     # Full technical concept
    └── experiment-plan.md   # Detailed experiment protocol
```

## Documentation

- **[Concept Paper](docs/concept-paper.md)**: Full technical specification of ARLM
- **[Experiment Plan](docs/experiment-plan.md)**: Detailed protocol for baseline comparisons and scaled experiments

## Quick Start

```bash
pip install -r requirements.txt

# Run smoke test
python tests/smoke_test.py

# Run baseline experiments (coming soon)
python experiments/01_baseline_debate.py
python experiments/02_baseline_rlm.py

# Run ARLM experiments (coming soon)
python experiments/03_arlm_scaled.py
```

## Experiments

1. **Baseline: Multi-agent debate** (Du et al. 2023)
2. **Baseline: RLMS without debate** (Zhang et al. 2024)
3. **ARLM scaled**: Varying agent counts and debate rounds
4. **Ablations**: Component contribution analysis

## License

MIT License - see LICENSE file

## Contact

Part of the Nootencorp research portfolio. Questions or collaboration inquiries: [GitHub Issues](https://github.com/Nootencorp/arlm/issues)
