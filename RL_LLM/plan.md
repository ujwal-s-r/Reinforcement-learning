REINFORCEMENT LEARNING FOR LLMS & AGENTS (Modules 1–5)
│
├── Module 1: RL Foundations for Language Models
│   ├── Lesson 1.1: The MDP Framing for Autoregressive Generation (States, Actions, Transitions, Rollouts)
│   ├── Lesson 1.2: Returns, Credit Assignment, and Reward Dynamics (Sparse vs. Dense, Shaping)
│   ├── Lesson 1.3: Policy Gradients & On-Policy Dynamics (REINFORCE intuition to PPO)
│   └── Lesson 1.4: Modern LLM RL Paradigm (KL Penalties, GRPO Mechanics, RLVR, and Reward Hacking)
│
├── Module 2: Practical LLM RL with TRL & GRPO
│   ├── Lesson 2.1: Alignment Landscape (SFT vs. DPO vs. PPO vs. GRPO — Mechanics & Tradeoffs)
│   ├── Lesson 2.2: Prompt Datasets, Grouped Rollouts & Relative Advantage Estimation
│   ├── Lesson 2.3: Deterministic & Rule-Based Reward Functions (Design, Pitfalls, Verification)
│   └── Lesson 2.4: Training Dynamics & Evaluation (Base vs. SFT vs. GRPO, Logging, Metric Interpretation)
│
├── Module 3: OpenEnv Essentials (Agent-Environment Loop)
│   ├── Lesson 3.1: Environment Architecture (`reset()`, `step()`, Observation/Action Schemas)
│   ├── Lesson 3.2: Trajectory Mechanics, Multi-Step State Transitions & Termination Criteria
│   ├── Lesson 3.3: Environment Execution Runtimes (Local, Subprocess, Sandboxed/Docker Environments)
│   └── Lesson 3.4: Trajectory Inspection, Parsing, and Action Validation
│
├── Module 4: OpenEnv + GRPO Integration (Multi-Turn RL)
│   ├── Lesson 4.1: Bridging TRL/GRPO with Interactive Environments (Stateful Rollout Pipelines)
│   ├── Lesson 4.2: Structured Tool & Action Parsing (Handling Malformed Outputs & Syntax Failures)
│   ├── Lesson 4.3: Environment Rewards vs. Model Judges (Signal Quality, Latency & Failure Modes)
│   └── Lesson 4.4: Reproducibility, Evaluation Seeds & End-to-End Multi-Turn Training Runs
│
├── Module 5: Production RL Project Polish & Evaluation
│   ├── Lesson 5.1: Rigorous Baseline Suite (Zero-Shot, Few-Shot SFT, GRPO on Held-Out Scenarios)
│   ├── Lesson 5.2: Reward Component Decomposition & Diagnostic Metrics
│   ├── Lesson 5.3: Failure Mode Taxonomy (Reward Gaming, Collapse, Tool Drift)
│   └── Lesson 5.4: Packaging, Dockerizing & Artifact Documentation for the Research Portfolio
│
└── Capstone Artifact:
    └── Reproducible, Dockerized Multi-Turn Environment + GRPO Trained LLM Agent with Full Evaluation Suite