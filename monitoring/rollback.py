import mlflow
import argparse
import sys
import os

def rollback_model(run_id, model_name="dqn_scheduler_policy"):
    """
    Simulates an automated rollback mechanism by loading a specific stable MLflow run 
    and overriding the current production model artifact.
    """
    print(f"⚠️  Initiating Rollback for Model '{model_name}' to Run ID: {run_id}")
    
    # In a real cluster, this would re-tag the model registry alias to @stable or @v1
    # For this project setup, we fetch the artifact from the specific run and replace the local deployed model
    
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    
    try:
        # Fetch the model from the specific historical run
        model_uri = f"runs:/{run_id}/{model_name}"
        stable_model = mlflow.pytorch.load_model(model_uri)
        
        print(f"✅ Successfully loaded stable model from MLflow (Run {run_id})")
        
        # Override the current production model in the 'experiments' folder
        import torch
        save_path = "experiments/saved_models/dqn_policy_final.pth"
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(stable_model.state_dict(), save_path)
        
        print(f"✅ Rollback complete. '{save_path}' has been overwritten with the stable policy.")
        print("🔄 Please restart the Docker / FastAPI service to load the restored model.")
        
    except Exception as e:
        print(f"❌ Rollback failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rollback production model to a previous MLflow run.")
    parser.add_argument("--run-id", required=True, help="The MLflow Run ID of the stable model to revert to.")
    args = parser.parse_args()
    
    rollback_model(args.run_id)
