import gymnasium as gym
from gymnasium import spaces
import numpy as np

class ProbeEnv(gym.Env):
    """
    Deep-Space Probe Communication Scheduling Environment
    """
    def __init__(self, max_steps=200):
        super(ProbeEnv, self).__init__()
        
        self.max_steps = max_steps
        self.current_step = 0
        
        # State space bounds (for normalization)
        self.max_battery = 100.0
        self.max_storage = 100.0
        self.max_queue = 50.0
        self.max_health = 100.0
        
        # Actions: 
        # 0: Transmit high-priority data
        # 1: Transmit low-priority data
        # 2: Conserve power
        # 3: Enter low-power mode
        self.action_space = spaces.Discrete(4)
        
        # Observation space: 7 variables
        # Normalized between 0 and 1
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(7,), dtype=np.float32
        )
        
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        # Initialize state variables randomly within reasonable bounds
        self.battery_level = np.random.uniform(70.0, 100.0)
        self.signal_strength = np.random.uniform(0.1, 1.0)
        self.storage_usage = np.random.uniform(10.0, 30.0)
        self.high_priority_queue = np.random.randint(0, 10)
        self.low_priority_queue = np.random.randint(0, 10)
        self.communication_window = np.random.choice([0, 1], p=[0.4, 0.6])
        self.system_health = 100.0
        
        return self._get_obs(), {}
        
    def _get_obs(self):
        obs = np.array([
            self.battery_level / self.max_battery,
            self.signal_strength,
            self.storage_usage / self.max_storage,
            self.high_priority_queue / self.max_queue,
            self.low_priority_queue / self.max_queue,
            float(self.communication_window),
            self.system_health / self.max_health
        ], dtype=np.float32)
        return np.clip(obs, 0.0, 1.0)
        
    def step(self, action):
        self.current_step += 1
        reward = 0.0
        terminated = False
        truncated = False
        
        # 1. Action Execution and Rewards
        if action == 0:  # Transmit high-priority
            if self.communication_window == 1 and self.battery_level >= 5 and self.high_priority_queue > 0:
                success_prob = self.signal_strength
                if np.random.random() < success_prob:
                    self.high_priority_queue -= 1
                    self.battery_level -= 5
                    self.storage_usage = max(0, self.storage_usage - 2)
                    reward += 15  # Successful high-priority
                else:
                    self.battery_level -= 5
                    reward -= 6   # Failed transmission
            else:
                reward -= 6  # Failed transmission (no window, no battery, or empty queue)
                self.battery_level -= 1 # Base cost
                
        elif action == 1:  # Transmit low-priority
            if self.communication_window == 1 and self.battery_level >= 3 and self.low_priority_queue > 0:
                success_prob = self.signal_strength
                if np.random.random() < success_prob:
                    self.low_priority_queue -= 1
                    self.battery_level -= 3
                    self.storage_usage = max(0, self.storage_usage - 1)
                    reward += 8  # Successful low-priority
                else:
                    self.battery_level -= 3
                    reward -= 6   # Failed transmission
            else:
                reward -= 6
                self.battery_level -= 1
                
        elif action == 2:  # Conserve power
            self.battery_level -= 0.5
            reward += 5
            if self.communication_window == 1 and (self.high_priority_queue > 0 or self.low_priority_queue > 0):
                reward -= 5  # Wasting communication window
                
        elif action == 3:  # Enter low-power mode
            self.battery_level -= 0.1
            self.system_health = min(100.0, self.system_health + 1)
            reward += 5
            if self.communication_window == 1 and (self.high_priority_queue > 0 or self.low_priority_queue > 0):
                reward -= 5
                
        # Maintaining healthy battery
        if self.battery_level > 50:
            reward += 3
            
        # 2. Environment Dynamics (Next Step Setup)
        
        # New packets arrive
        if np.random.random() < 0.3:
            self.high_priority_queue += np.random.randint(1, 4)
            self.storage_usage += 2
        if np.random.random() < 0.5:
            self.low_priority_queue += np.random.randint(1, 5)
            self.storage_usage += 1
            
        # Update comm window and signal
        self.communication_window = np.random.choice([0, 1], p=[0.4, 0.6])
        if self.communication_window == 1:
            self.signal_strength = np.random.uniform(0.1, 1.0)
        else:
            self.signal_strength = 0.0
            
        # Passive drain
        self.battery_level -= 0.2
        
        # Constraints
        self.high_priority_queue = min(self.high_priority_queue, self.max_queue)
        self.low_priority_queue = min(self.low_priority_queue, self.max_queue)
        
        # 3. Termination Conditions
        if self.storage_usage >= 100:
            reward -= 12
            terminated = True
            
        if self.battery_level <= 0:
            reward -= 25
            terminated = True
            
        if self.system_health <= 0:
            terminated = True
            
        if self.current_step >= self.max_steps:
            truncated = True
            
        return self._get_obs(), reward, terminated, truncated, {}
