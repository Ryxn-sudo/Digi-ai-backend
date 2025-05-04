import os
import json
import sys
from datetime import datetime
import subprocess

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import TRAINING_DATA_CONFIG

def check_contribution_count():
    """Check if we have enough contributions for retraining"""
    status_file = TRAINING_DATA_CONFIG['STATUS_FILE']
    
    if not os.path.exists(status_file):
        print("Status file not found. Please ensure the server has been run at least once.")
        return False
    
    try:
        with open(status_file, 'r') as f:
            status = json.load(f)
        
        unprocessed = status.get('unprocessed_contributions', 0)
        target = status.get('next_retrain_target', TRAINING_DATA_CONFIG['MIN_SAMPLES_FOR_RETRAINING'])
        
        print(f"Current unprocessed contributions: {unprocessed}/{target}")
        
        return unprocessed >= target
    except Exception as e:
        print(f"Error checking contribution count: {e}")
        return False

if __name__ == "__main__":
    print(f"Checking contribution status at {datetime.now().isoformat()}")
    
    if check_contribution_count():
        print("Enough contributions available for retraining. Starting retraining process...")
        # Call the retrain_model.py script
        try:
            script_path = os.path.join(os.path.dirname(__file__), "retrain_model.py")
            subprocess.run([sys.executable, script_path], check=True)
            print("Retraining process completed.")
        except subprocess.CalledProcessError as e:
            print(f"Retraining failed with error code {e.returncode}")
    else:
        print("Not enough contributions for retraining yet.")
