import os
import sys
import yaml
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.dqn_agent import DQNAgent

app = FastAPI(title="Deep Space Probe Scheduling API")

class ProbeState(BaseModel):
    battery_level: float
    signal_strength: float
    storage_usage: float
    high_priority_queue: int
    low_priority_queue: int
    communication_window: int
    system_health: float

# Load agent at startup
try:
    with open('configs/dqn_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    agent = DQNAgent(state_dim=7, action_dim=4, config=config)
    
    model_path = 'experiments/saved_models/dqn_policy_final.pth'
    if os.path.exists(model_path):
        agent.load(model_path)
    else:
        print("Warning: Trained model not found. Using untrained policy.")
except Exception as e:
    print(f"Initialization error: {e}")
    agent = None

action_mapping = {
    0: "transmit_high_priority",
    1: "transmit_low_priority",
    2: "conserve_power",
    3: "enter_low_power_mode"
}

@app.get("/status")
def get_status():
    return {
        "status": "running",
        "model": "DQN Scheduler",
        "version": "1.0"
    }

@app.post("/schedule")
def schedule(state: ProbeState):
    if agent is None:
        return {"error": "Agent not initialized"}
        
    # Normalize state as per environment max bounds
    max_battery = 100.0
    max_storage = 100.0
    max_queue = 50.0
    max_health = 100.0
    
    normalized_state = np.array([
        state.battery_level / max_battery,
        state.signal_strength,
        state.storage_usage / max_storage,
        state.high_priority_queue / max_queue,
        state.low_priority_queue / max_queue,
        float(state.communication_window),
        state.system_health / max_health
    ], dtype=np.float32)
    
    normalized_state = np.clip(normalized_state, 0.0, 1.0)
    
    action_id = agent.select_action(normalized_state, evaluate=True)
    recommended_action = action_mapping.get(action_id, "unknown")
    
    return {
        "recommended_action": recommended_action
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
