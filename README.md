# Deep Space Probe Communication Scheduling using Deep Q-Networks (DQN) and MLOps

## 1. Project Overview
This project implements an intelligent communication scheduling system for a simulated deep-space probe using Deep Reinforcement Learning (DQN) with integrated MLOps practices. 

Deep-space probes operate in highly resource-constrained environments where communication opportunities are limited and mission resources must be utilized efficiently. This project uses a DQN agent to autonomously learn how to prioritize communication tasks, optimize transmission decisions, conserve probe energy, and maximize mission efficiency.

## 2. Setup and Installation
Install the necessary dependencies using pip:

```bash
pip install -r requirements.txt
```

## 3. Training the DQN Agent
Run the training pipeline which tracks experiments using MLflow and saves the metrics and policies.

```bash
python training/train.py --config configs/dqn_config.yaml
```

Metrics will be saved to `experiments/metrics/training_metrics.csv` and the model to `experiments/saved_models/dqn_policy_final.pth`.

## 4. Evaluation
Compare the trained DQN agent with the rule-based baseline scheduler and generate visualizations.

```bash
python evaluation/evaluate.py
```

This will produce comparison tables and save plots in the `plots/` directory and comparison metrics in the `results/` directory.

## 5. Monitoring
To simulate the monitoring of probe operational metrics:

```bash
python monitoring/monitor.py
```
This logs operational metrics, failed transmissions, and resource overloads into `experiments/logs/monitoring.log`.

## 6. FastAPI Service
A deployment-ready API provides an interface for interacting with the trained model.

Start the FastAPI server:
```bash
python api/api.py
```

### Endpoints
- **GET /status**: Check system status.
- **POST /schedule**: Request an optimal action for a given state.

Example POST request:
```json
{
  "battery_level": 75,
  "signal_strength": 0.82,
  "storage_usage": 55,
  "high_priority_queue": 12,
  "low_priority_queue": 8,
  "communication_window": 1,
  "system_health": 90
}
```

## 7. Project Architecture
```text
DeepSpace_RL_MLOps/
├── configs/
│   └── dqn_config.yaml
├── env/
│   └── probe_env.py
├── agents/
│   ├── dqn_agent.py
│   └── replay_buffer.py
├── models/
│   └── dqn_network.py
├── training/
│   └── train.py
├── evaluation/
│   └── evaluate.py
├── baseline/
│   └── baseline_scheduler.py
├── monitoring/
│   └── monitor.py
├── api/
│   └── api.py
├── experiments/
│   ├── logs/
│   ├── metrics/
│   └── saved_models/
├── plots/
├── results/
├── requirements.txt
└── README.md
```
