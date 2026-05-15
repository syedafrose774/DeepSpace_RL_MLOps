# Problem Analysis & Design Requirements
**Project:** Deep Space Probe Communication Scheduling using Deep Q-Networks (DQN)

## 1. Stakeholders
- **Mission Control Operators:** Engineers responsible for monitoring probe health and issuing manual overrides if necessary.
- **Space Agencies (e.g., ISRO, NASA):** The primary beneficiaries requiring maximum scientific data throughput and mission longevity.
- **Scientific Researchers:** End-users relying on the uncorrupted delivery of high-priority telemetry and experimental data.

## 2. Use Cases
1. **Autonomous Scheduling (Primary):** The RL agent automatically evaluates the 7-dimensional telemetry state and selects the optimal communication mode (transmit, conserve, sleep) without human intervention.
2. **Health Monitoring:** The system continuously logs battery depletion rates and transmission failures, alerting Mission Control of anomalies.
3. **Data Queue Management:** The system dynamically categorizes and prioritizes scientific data against routine telemetry to prevent buffer overflows.

## 3. Functional Requirements
- **FR1 (State Observation):** The system must accurately observe the 7-variable environment state at each timestep.
- **FR2 (Action Execution):** The RL agent must output one of four discrete actions to the environment.
- **FR3 (Experiment Tracking):** All training metrics (reward, loss, epsilon) must be logged persistently using MLflow.
- **FR4 (API Serving):** The trained model must be exposed via a RESTful API (FastAPI) capable of processing JSON telemetry payloads.

## 4. Non-Functional Requirements
- **NFR1 (Latency):** The FastAPI inference endpoint must return an action in under 100 milliseconds to support real-time operations.
- **NFR2 (Reliability):** The system must prevent the probe battery from reaching 0% under standard stochastic conditions.
- **NFR3 (Scalability):** The entire deployment environment must be containerized (Docker) to ensure identical execution across different edge devices.
- **NFR4 (Reproducibility):** Hyperparameters and model versions must be versioned and reproducible via Git and MLflow.

## 5. Feasibility
- **Technical Feasibility:** Highly feasible. Using a lightweight PyTorch neural network allows inference on resource-constrained hardware, while FastAPI and Docker provide a reliable MLOps wrapper.
- **Operational Feasibility:** Automating the scheduling significantly reduces the cognitive load on human operators, increasing overall mission safety.

## 6. Constraints
- **C1 (Energy Limits):** The probe operates strictly on limited battery power that replenishes slowly.
- **C2 (Storage Limits):** Onboard solid-state memory has a strict maximum capacity.
- **C3 (Communication Windows):** Transmission is only physically possible during specific orbital alignments with Earth.

## 7. Trade-Offs
- **Throughput vs. Survival:** The system actively trades off short-term data transmission throughput in favor of long-term battery conservation, ensuring the hardware survives periods of weak cosmic signal.
- **Exploration vs. Exploitation:** During training, short-term rewards are sacrificed (via $\epsilon$-greedy exploration) to discover long-term optimal survival policies.

## 8. Risks
- **R1 (Concept Drift):** The space environment may undergo unprecedented changes (e.g., an extreme solar flare) causing the data distribution to shift away from the training distribution.
- **R2 (Hardware Failure):** Physical degradation of the transmitter is not currently simulated but poses a real-world risk.

## 9. Traceability Matrix
| Requirement ID | Description | Implementation Component | Testing/Verification |
| :--- | :--- | :--- | :--- |
| FR3 | MLflow Tracking | `train.py`, MLflow UI | Visual verification via `http://localhost:5000` |
| FR4 | API Serving | `api/api.py` (FastAPI) | `curl -X POST` to `/schedule` endpoint |
| NFR1 | Latency | `evaluate.py` | Profiling average API response time |
| NFR3 | Containerization | `Dockerfile`, `docker-compose.yml`| Automated GitHub Actions CI pipeline |
