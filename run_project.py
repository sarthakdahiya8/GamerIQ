import subprocess
import sys
import os

def setup_project():
    """Complete project setup"""
    print("🎮 Setting up GamerIQ Project...")
    
    # Create directories
    directories = ['data/datasets', 'models', 'api', 'frontend']
    for dir_name in directories:
        os.makedirs(dir_name, exist_ok=True)
        print(f"✅ Created directory: {dir_name}")
    
    # Generate synthetic data
    print("📊 Generating synthetic gaming data...")
    os.system("python data/synthetic_game_data.py")
    
    # Train models
    print("🤖 Training ML models...")
    os.system("python models/player_churn_prediction.py")
    
    print("🚀 Project setup complete!")
    print("\nTo run the project:")
    print("1. Dashboard: streamlit run api/dashboard.py")
    print("2. API Server: python api/flask_api.py")

if __name__ == "__main__":
    setup_project()
