"""
config.py - Experiment configurations for ARLM

Defines parameters for baseline comparisons and scaled experiments.
"""

# Baseline configurations
BASELINE_DEBATE_CONFIG = {
    "num_agents": 3,
    "num_rounds": 2,
    "model": "gpt-4",
    "temperature": 0.7
}

BASELINE_RLM_CONFIG = {
    "num_iterations": 100,
    "batch_size": 32,
    "learning_rate": 1e-4
}

# ARLM scaled experiment
ARLM_SCALED_CONFIG = {
    "num_agents": [3, 5, 10],  # Scale agent pool
    "num_rounds": [2, 3, 5],   # Scale debate depth
    "rlms_iterations": 200,
    "model": "gpt-4",
    "temperature": 0.7
}

# Ablation study configurations
ABLATION_CONFIGS = {
    "no_debate": {"use_debate": False, "use_rlms": True},
    "no_rlms": {"use_debate": True, "use_rlms": False},
    "full_arlm": {"use_debate": True, "use_rlms": True}
}
