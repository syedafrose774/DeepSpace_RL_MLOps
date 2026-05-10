import os
import sys
import yaml
import mlflow
import argparse
import numpy as np
import pandas as pd
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.probe_env import ProbeEnv
from agents.dqn_agent import DQNAgent

def train(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        
    env = ProbeEnv(max_steps=config['max_steps_per_episode'])
    agent = DQNAgent(state_dim=7, action_dim=4, config=config)
    
    os.makedirs('experiments/saved_models', exist_ok=True)
    os.makedirs('experiments/metrics', exist_ok=True)
    
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("DeepSpace_DQN_Scheduling")
    
    with mlflow.start_run():
        mlflow.log_params(config)
        
        episodes = config['episodes']
        rewards = []
        losses = []
        transmission_success_rates = []
        battery_remainings = []
        storage_overflows = []
        
        for episode in range(episodes):
            state, _ = env.reset()
            total_reward = 0
            episode_losses = []
            
            successful_transmissions = 0
            total_transmission_attempts = 0
            storage_overflow_count = 0
            
            done = False
            
            while not done:
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Metrics logic
                if action in [0, 1]:
                    total_transmission_attempts += 1
                    if reward == 15 or reward == 8:
                        successful_transmissions += 1
                        
                if env.storage_usage >= 100:
                    storage_overflow_count += 1
                
                agent.store_transition(state, action, reward, next_state, done)
                
                loss = agent.optimize_model()
                if loss is not None:
                    episode_losses.append(loss)
                    
                state = next_state
                total_reward += reward
                
            agent.steps_done += 1
            if episode % agent.target_update_frequency == 0:
                agent.update_target_network()
                
            # Log metrics per episode
            avg_loss = np.mean(episode_losses) if episode_losses else 0
            success_rate = successful_transmissions / max(1, total_transmission_attempts)
            
            rewards.append(total_reward)
            losses.append(avg_loss)
            transmission_success_rates.append(success_rate)
            battery_remainings.append(env.battery_level)
            storage_overflows.append(storage_overflow_count)
            
            mlflow.log_metric("episode_reward", total_reward, step=episode)
            mlflow.log_metric("episode_loss", avg_loss, step=episode)
            mlflow.log_metric("epsilon", agent.epsilon, step=episode)
            mlflow.log_metric("success_rate", success_rate, step=episode)
            mlflow.log_metric("battery_remaining", env.battery_level, step=episode)
            
            if (episode + 1) % 50 == 0:
                print(f"Episode {episode + 1}/{episodes} | Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.2f} | Loss: {avg_loss:.4f}")
                
        # Save model
        model_path = f"experiments/saved_models/dqn_policy_final.pth"
        agent.save(model_path)
        mlflow.log_artifact(model_path)
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame({
            'episode': range(1, episodes + 1),
            'reward': rewards,
            'loss': losses,
            'transmission_success_rate': transmission_success_rates,
            'battery_remaining': battery_remainings,
            'storage_overflow_count': storage_overflows
        })
        metrics_csv_path = 'experiments/metrics/training_metrics.csv'
        metrics_df.to_csv(metrics_csv_path, index=False)
        mlflow.log_artifact(metrics_csv_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/dqn_config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    train(args.config)
