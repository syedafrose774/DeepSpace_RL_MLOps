import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.probe_env import ProbeEnv
from agents.dqn_agent import DQNAgent
from baseline.baseline_scheduler import BaselineScheduler

def evaluate_agent(env, agent, num_episodes=50, is_dqn=True):
    metrics = {
        'rewards': [],
        'success_rates': [],
        'battery_remaining': [],
        'storage_overflows': [],
        'episode_lengths': []
    }
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        successful_transmissions = 0
        total_transmissions = 0
        overflows = 0
        steps = 0
        
        done = False
        while not done:
            if is_dqn:
                action = agent.select_action(state, evaluate=True)
            else:
                action = agent.select_action(state)
                
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            if action in [0, 1]:
                total_transmissions += 1
                if reward in [15, 8]:
                    successful_transmissions += 1
                    
            if env.storage_usage >= 100:
                overflows += 1
                
            total_reward += reward
            state = next_state
            steps += 1
            
        metrics['rewards'].append(total_reward)
        metrics['success_rates'].append(successful_transmissions / max(1, total_transmissions))
        metrics['battery_remaining'].append(env.battery_level)
        metrics['storage_overflows'].append(overflows)
        metrics['episode_lengths'].append(steps)
        
    return {k: np.mean(v) for k, v in metrics.items()}

def plot_training_metrics(metrics_csv='experiments/metrics/training_metrics.csv'):
    if not os.path.exists(metrics_csv):
        print(f"Metrics file {metrics_csv} not found.")
        return
        
    df = pd.read_csv(metrics_csv)
    os.makedirs('plots', exist_ok=True)
    
    # 1. Reward vs Episodes
    plt.figure(figsize=(10, 6))
    plt.plot(df['episode'], df['reward'], label='Reward', color='blue')
    plt.title('DQN Training: Reward vs Episodes')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.grid()
    plt.savefig('plots/reward_vs_episodes.png')
    plt.close()
    
    # 2. Transmission Success Rate
    plt.figure(figsize=(10, 6))
    plt.plot(df['episode'], df['transmission_success_rate'], color='green')
    plt.title('DQN Training: Transmission Success Rate')
    plt.xlabel('Episodes')
    plt.ylabel('Success Rate')
    plt.grid()
    plt.savefig('plots/transmission_success_rate.png')
    plt.close()

    # 3. Battery Usage Trends
    plt.figure(figsize=(10, 6))
    plt.plot(df['episode'], df['battery_remaining'], color='orange')
    plt.title('DQN Training: Remaining Battery')
    plt.xlabel('Episodes')
    plt.ylabel('Battery Level')
    plt.grid()
    plt.savefig('plots/battery_usage_trends.png')
    plt.close()

    # 4. Storage Utilization
    plt.figure(figsize=(10, 6))
    plt.plot(df['episode'], df['storage_overflow_count'], color='red')
    plt.title('DQN Training: Storage Overflow Count')
    plt.xlabel('Episodes')
    plt.ylabel('Overflow Count')
    plt.grid()
    plt.savefig('plots/storage_utilization.png')
    plt.close()

def run_evaluation(config_path, model_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    env = ProbeEnv(max_steps=config['max_steps_per_episode'])
    
    # Init and load DQN Agent
    dqn_agent = DQNAgent(state_dim=7, action_dim=4, config=config)
    if os.path.exists(model_path):
        dqn_agent.load(model_path)
    else:
        print(f"Warning: Model not found at {model_path}. Using random initialization.")
        
    baseline_agent = BaselineScheduler()
    
    print("Evaluating DQN Agent...")
    dqn_metrics = evaluate_agent(env, dqn_agent, num_episodes=50, is_dqn=True)
    
    print("Evaluating Baseline Scheduler...")
    baseline_metrics = evaluate_agent(env, baseline_agent, num_episodes=50, is_dqn=False)
    
    # Save comparison to results directory
    os.makedirs('results', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    comparison_df = pd.DataFrame([baseline_metrics, dqn_metrics], index=['Baseline', 'DQN'])
    comparison_df.to_csv('results/comparison_metrics.csv')
    
    print("\n--- Final Comparison Table ---")
    print(comparison_df.to_markdown())
    
    # 5. Baseline vs DQN Comparison Plot
    comparison_df[['rewards']].plot(kind='bar', figsize=(8, 5), title='Average Reward Comparison', color=['cyan', 'magenta'])
    plt.ylabel('Average Reward')
    plt.tight_layout()
    plt.savefig('plots/baseline_vs_dqn_reward.png')
    plt.close()
    
    plot_training_metrics()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/dqn_config.yaml")
    parser.add_argument("--model", type=str, default="experiments/saved_models/dqn_policy_final.pth")
    args = parser.parse_args()
    
    run_evaluation(args.config, args.model)
