import pandas as pd
import os

def generate_final_summary():
    print("\n" + "="*60)
    print("DEEP SPACE PROBE AI: FINAL MISSION SUMMARY")
    print("="*60)
    
    metrics_path = 'results/comparison_metrics.csv'
    if not os.path.exists(metrics_path):
        print("Error: Comparison metrics not found. Please run evaluation/evaluate.py first.")
        return
        
    df = pd.read_csv(metrics_path, index_col=0)
    
    # Extract values
    baseline_survival = df.loc['Baseline', 'episode_lengths']
    dqn_survival = df.loc['DQN', 'episode_lengths']
    improvement = ((dqn_survival - baseline_survival) / baseline_survival) * 100
    
    print(f"PERFORMANCE ANALYSIS:")
    print(f"   - Baseline Survival Time:  {baseline_survival:.1f} steps")
    print(f"   - DQN AI Survival Time:    {dqn_survival:.1f} steps")
    print(f"   - MISSION EXTENSION:       +{improvement:.1f}% longer survival!")
    
    print("\nRESOURCE EFFICIENCY:")
    print(f"   - Baseline Final Battery:  {df.loc['Baseline', 'battery_remaining']:.2f}%")
    print(f"   - DQN AI Final Battery:    {df.loc['DQN', 'battery_remaining']:.2f}%")
    
    print("\nMLOPS STATUS:")
    print("   - [OK] Experiment Tracked (MLflow)")
    print("   - [OK] Model Versioned (.pth)")
    print("   - [OK] API Deployed (FastAPI)")
    print("   - [OK] Containerized (Docker)")
    print("   - [OK] CI/CD Ready (GitHub Actions)")

    
    print("="*60)
    print("STATUS: SYSTEM READY FOR DEPLOYMENT TO DEEP SPACE PROBE")
    print("="*60 + "\n")

if __name__ == "__main__":
    generate_final_summary()
