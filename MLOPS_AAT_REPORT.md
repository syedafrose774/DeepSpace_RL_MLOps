# MLOPS AAT REPORT: End-to-End Production Pipeline for Autonomous Space Probes

**Project Title:** End-to-End MLOps Pipeline for Deep Space RL Agents
**Course:** Machine Learning Operations (24AM6AEMLO)
**Department:** Machine Learning / Artificial Intelligence
**Team Members:** [YOUR NAME] ([YOUR USN]), [FRIEND NAME] ([FRIEND USN])

---

## 1. INTRODUCTION & PROJECT GOALS
In the modern aerospace industry, developing a mathematical AI model is only a fraction of the engineering challenge. The primary bottleneck is deploying, scaling, and monitoring that model in a reliable production environment. This project aims to bridge the gap between theoretical Reinforcement Learning (RL) and production-grade software engineering by designing a comprehensive **Machine Learning Operations (MLOps)** pipeline. 

The core objective is to take a Deep Q-Network (DQN) scheduling agent—designed for a deep space probe—and wrap it in a robust ecosystem that includes automated experiment tracking, model versioning, continuous integration (CI/CD), containerization, and a real-time monitoring web dashboard.

---

## 2. SUSTAINABLE DEVELOPMENT GOALS (SDG) ALIGNMENT
This project strongly adheres to the United Nations SDGs:
- **SDG 9 (Industry, Innovation, and Infrastructure):** By building automated deployment pipelines and intelligent APIs, we are advancing the state of resilient digital infrastructure for the aerospace sector. The automation reduces human operational overhead and increases system reliability.
- **SDG 12 (Responsible Consumption and Production):** Spacecraft possess severely limited physical resources (battery lifespan and solid-state storage). Our ML model directly optimizes the consumption of these resources, ensuring that the probe operates efficiently and reduces electronic wear, thereby extending its functional lifecycle.

---

## 3. MLOPS SYSTEM ARCHITECTURE & LIFECYCLE
We designed a 4-tier MLOps architecture that strictly separates the concerns of Data/Training, Tracking, Deployment, and Monitoring.

### 3.1 Tier 1: Experiment Tracking & Model Registry (MLflow)
Training Reinforcement Learning models involves significant hyperparameter tuning. To eliminate manual record-keeping, we integrated **MLflow** directly into the PyTorch training loop.
- **Parameter Logging:** Every run automatically logs the Learning Rate, Discount Factor ($\gamma$), Target Network Update Frequency, and Epsilon Decay rates.
- **Metric Tracking:** MLflow tracks the Episode Reward, Loss, and Battery Remaining at each step, allowing us to visualize convergence over 500 episodes in the MLflow UI (`http://localhost:5000`).
- **Artifact Registry:** Instead of losing models in local folders, the final converged PyTorch model (`dqn_policy_final.pth`) is saved as a versioned artifact within MLflow, ensuring absolute reproducibility.

### 3.2 Tier 2: Continuous Integration & Deployment (CI/CD)
To guarantee that new code does not break the AI agent, we implemented a GitOps workflow using **GitHub Actions**.
- A `.github/workflows/ci.yml` pipeline triggers on every push to the `main` branch.
- It provisions an Ubuntu runner, installs dependencies via `requirements.txt`, and executes both the `train.py` and `evaluate.py` scripts. 
- This automated testing ensures that the PyTorch environment compiles correctly and that the agent's baseline evaluation metrics do not regress before deployment.

### 3.3 Tier 3: Containerization & Portability (Docker)
The entire inference architecture is decoupled from the host operating system using Docker.
- We crafted a `Dockerfile` utilizing a lightweight `python:3.11-slim` base image.
- A `docker-compose.yml` file is provided to orchestrate the deployment of the API and its dependencies with a single command (`docker-compose up`). This ensures parity between the developer's laptop and the final production server.

### 3.4 Tier 4: Model Serving (FastAPI) & Frontend Integration
The trained model is not left as a static script; it is deployed as an active microservice.
- **FastAPI Framework:** We utilize FastAPI (running on an ASGI Uvicorn server) to expose the model via a REST API. It is highly performant and provides automatic OpenAPI (Swagger) documentation at `/docs`.
- **The `/schedule` Endpoint:** This POST endpoint accepts a JSON payload of the probe's current 7-dimensional telemetry. It loads the `dqn_policy_final.pth` model into memory, performs a forward pass, and returns the mathematically optimal action (e.g., `{"recommended_action": "conserve_power"}`).
- **Interactive UI Dashboard:** We developed a responsive, glassmorphism-themed web interface (HTML/CSS/JS) that interacts with the FastAPI backend via asynchronous `fetch()` calls. It includes manual sliders and an "Auto-Pilot" simulation mode for live stakeholder demonstrations.

---

## 4. PRODUCTION MONITORING & DRIFT DETECTION
A deployed ML model can degrade if the live production data distribution shifts away from the training data distribution (Data Drift). We implemented a production monitoring script (`monitoring/monitor.py`) to safeguard the system.
- **Battery Depletion Alerts:** The monitor tracks the rate of battery loss. If the probe consumes energy faster than the expected baseline (indicating a shift in cosmic weather or a model degradation), it logs a `CRITICAL` alert.
- **Transmission Failure Tracking:** If the agent begins making sub-optimal decisions (e.g., attempting transmissions during continuous weak signals), the monitor tracks the ratio of failed packets and triggers warnings for mission control.

---

## 5. EXPERIMENTAL RESULTS & DEPLOYMENT METRICS
By utilizing our MLOps evaluation scripts, we extracted the following metrics proving the efficacy of our deployment:
- **Baseline Average Reward:** 142.26
- **RL Model Average Reward:** 406.38
- **Survival Extension:** The containerized RL agent extended mission survival from 28 timesteps to 87 timesteps on average.
- **API Latency:** The FastAPI `/schedule` endpoint processes the 7-dimensional state tensor and returns an action in under 15 milliseconds, proving it is capable of real-time edge deployment.

---

## 6. CONCLUSION
This project successfully demonstrates the immense value of Machine Learning Operations. By wrapping a complex Deep Reinforcement Learning algorithm in a rigorous CI/CD, Containerization, and API tracking pipeline, we transformed a fragile scientific script into resilient, production-ready aerospace software. The system is scalable, reproducible, and actively monitored, fulfilling all requirements for modern MLOps architecture.

---

## 7. REFERENCES
1. MLflow Documentation: Experiment Tracking and Model Registry (https://mlflow.org/docs)
2. FastAPI Framework: High-performance ASGI serving (https://fastapi.tiangolo.com)
3. Docker Best Practices for Python Microservices.
4. Mnih, V., et al. "Human-level control through deep reinforcement learning." Nature 518.7540 (2015).
