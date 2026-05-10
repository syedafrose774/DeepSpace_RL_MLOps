import time
import random
import logging
import json
import os

os.makedirs('experiments/logs', exist_ok=True)

# Configure logging
logging.basicConfig(
    filename='experiments/logs/monitoring.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ProbeMonitor:
    def __init__(self):
        self.battery_drift_threshold = 20.0
        self.storage_limit = 90.0
        self.failed_transmissions = 0
        
    def check_health(self, state, action, reward):
        # state is normalized
        battery_level = state[0] * 100
        storage_usage = state[2] * 100
        
        alerts = []
        
        if battery_level < self.battery_drift_threshold:
            alerts.append(f"CRITICAL: Battery level dangerously low ({battery_level:.2f}%)")
            
        if storage_usage > self.storage_limit:
            alerts.append(f"WARNING: Storage approaching overload ({storage_usage:.2f}%)")
            
        if action in [0, 1] and reward < 0:
            self.failed_transmissions += 1
            alerts.append(f"ERROR: Transmission failed. Total failures: {self.failed_transmissions}")
            
        if reward < -10:
            alerts.append(f"WARNING: Abnormal reward drop detected (Reward: {reward})")
            
        for alert in alerts:
            if "CRITICAL" in alert or "ERROR" in alert:
                logging.error(alert)
                print(alert)
            else:
                logging.warning(alert)
                
        return alerts

if __name__ == "__main__":
    monitor = ProbeMonitor()
    print("Probe Monitoring System Started...")
    
    # Simulate a stream of events
    try:
        while True:
            # Simulated normalized state
            mock_state = [
                random.uniform(0.1, 0.9), # battery
                random.uniform(0.1, 1.0), # signal
                random.uniform(0.1, 0.95), # storage
                random.uniform(0.0, 1.0),
                random.uniform(0.0, 1.0),
                random.choice([0, 1]),
                random.uniform(0.5, 1.0)
            ]
            
            mock_action = random.choice([0, 1, 2, 3])
            
            # Simulate reward based on random action and state
            mock_reward = random.choice([15, 8, 5, 3, -6, -12, -25])
            
            monitor.check_health(mock_state, mock_action, mock_reward)
            
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("Monitoring System Stopped.")
