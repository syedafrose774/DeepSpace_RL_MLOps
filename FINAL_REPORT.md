# Final Project Report: Deep Space Probe Communication Scheduling

## 1. Problem Statement
Deep-space probes operate in highly resource-constrained environments. They must manage limited battery power, varying communication signal strength, and random communication windows with Earth, all while trying to transmit scientifically valuable data. Traditional rule-based programming drains battery quickly because it blindly tries to send data whenever a window is open, ignoring long-term survival. This project solves this by using Deep Reinforcement Learning (DQN) to act as an intelligent scheduling agent.

## 2. SDG Impact
This project directly supports **SDG 9 (Industry, Innovation and Infrastructure)** by improving the intelligence and efficiency of space technology infrastructure. Furthermore, it aligns with **SDG 12 (Responsible Consumption and Production)** by optimizing the consumption of the probe's critical resources (battery energy and storage capacity), significantly reducing waste.

## 3. Simulator (Environment)
The environment `env/probe_env.py` is a custom OpenAI Gym simulator. 
* **State Space:** 7 variables including battery level, signal strength, storage usage, high/low priority data queues, communication window availability, and system health.
* **Action Space:** 4 actions (Transmit high-priority, Transmit low-priority, Conserve power, Enter low-power mode).
* **Dynamics:** Every timestep drains battery passively. Random cosmic events generate new data packets.
* **Rewards:** Positive rewards for successful transmissions and power conservation. Large penalties for storage overflow (-12), transmission failure (-6), and battery depletion (-25).

## 4. Reinforcement Learning (RL) Methodology
We chose the **Deep Q-Network (DQN)** algorithm because the state variables (like battery percentage and signal strength) are continuous, making standard Q-tables impossible. 
* **Exploration:** We use an $\epsilon$-greedy strategy, decaying from 1.0 to 0.05.
* **Architecture:** A PyTorch neural network with 2 hidden layers (128 and 64 neurons) mapping the 7-dimensional state to 4 Q-values.
* **Training:** The model is trained over 500 episodes using an Experience Replay buffer and a Target Network to stabilize learning.

## 5. MLOps Implementation
The project follows strict MLOps principles:
* **Experiment Tracking & Registry:** We use `MLflow` to track hyperparameters (learning rate, epsilon), log episode metrics (reward, loss, battery remaining), and version our trained models (`dqn_policy_final.pth`).
* **CI/CD & Automation:** We implemented a GitHub Actions workflow (`.github/workflows/ci.yml`) to automatically test the pipeline on code pushes.
* **Containerization:** The inference API is containerized using `Dockerfile` and `docker-compose.yml` for scalable deployment.
* **Deployment API:** A FastAPI server (`api/api.py`) exposes the trained model via a `/schedule` REST endpoint.
* **Monitoring:** A logging script (`monitor.py`) checks for operational constraints and logs warnings if the battery drifts below thresholds or if excessive transmission failures occur.

## 6. Results and Analysis
The final evaluation over 50 episodes yielded the following performance comparison against a Baseline fixed-rule scheduler:

| Metric | Baseline | RL Policy (DQN) |
|--------|----------|-----------------|
| **Average Reward** | 119.0 | **357.3** |
| **Average Battery Remaining** | -0.54% | **59.18%** |
| **Episode Length (Survival)** | 27.3 steps | **73.0 steps** |

### Insights: When does RL perform better?
The RL policy vastly outperforms the Baseline when resources are low. While the Baseline attempts to blindly transmit data just because a communication window is open—leading to massive transmission failures if the signal is weak—the RL agent learns to skip weak signals and **conserve power**, allowing it to survive almost 3x longer and harvest significantly more overall reward.

### When does it behave unexpectedly?
The RL agent sometimes achieves a very low "transmission success rate" compared to what humans might expect. This isn't a failure; it's a learned survival tactic. It avoids transmitting unless conditions are absolutely perfect, which looks like "inactivity" but is actually hyper-optimized resource preservation.

## 7. Limitations and Future Work
* **Sensitivity to Traffic Patterns:** If the random generation rate of high-priority packets drastically increases beyond the training distribution, the RL agent's strict conservation policies might lead to storage overflows because it is too conservative to clear the queue in time.
* **Future Work:** Implementing a Proximal Policy Optimization (PPO) algorithm could provide more stable continuous control, and live drift-detection tools (like EvidentlyAI) could be added to the FastAPI service to monitor state-distribution shifts in real-time.
