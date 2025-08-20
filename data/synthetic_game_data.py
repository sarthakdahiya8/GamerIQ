import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_gaming_data(num_players=10000):
    """
    Gaming industry ke realistic data generate karta hai
    """
    np.random.seed(42)
    
    # Player demographics
    players = []
    
    for i in range(num_players):
        player_id = f"player_{i:06d}"
        
        # Demographics
        age = np.random.choice([16, 18, 22, 25, 28, 32, 35], p=[0.1, 0.15, 0.2, 0.2, 0.15, 0.1, 0.1])
        country = np.random.choice(['India', 'USA', 'UK', 'Brazil', 'Germany'], 
                                 p=[0.4, 0.2, 0.1, 0.15, 0.15])
        
        # Gaming behavior patterns
        skill_level = np.random.choice(['Beginner', 'Intermediate', 'Advanced', 'Pro'], 
                                     p=[0.3, 0.4, 0.25, 0.05])
        
        # Engagement metrics (realistic gaming patterns)
        daily_playtime = max(0.5, np.random.exponential(2.5))  # Hours per day
        sessions_per_week = max(1, int(np.random.poisson(8)))
        
        # Purchase behavior
        is_paying = np.random.choice([0, 1], p=[0.7, 0.3])
        monthly_spend = 0 if not is_paying else max(0, np.random.exponential(25))
        
        # Performance metrics
        win_rate = np.random.beta(2, 3) if skill_level == 'Beginner' else \
                  np.random.beta(3, 2) if skill_level == 'Intermediate' else \
                  np.random.beta(4, 1.5) if skill_level == 'Advanced' else \
                  np.random.beta(5, 1)
        
        # Social features
        friends_count = max(0, int(np.random.poisson(15)))
        guild_member = np.random.choice([0, 1], p=[0.4, 0.6])
        
        # Churn indicators
        days_since_last_login = max(0, int(np.random.exponential(3)))
        complaint_count = max(0, int(np.random.poisson(0.5)))
        
        # Will churn? (target variable)
        churn_probability = (
            (age > 30) * 0.1 +
            (daily_playtime < 1) * 0.3 +
            (monthly_spend == 0) * 0.2 +
            (days_since_last_login > 7) * 0.4 +
            (complaint_count > 2) * 0.3 +
            (friends_count < 5) * 0.1
        )
        will_churn = np.random.random() < min(churn_probability, 0.8)
        
        players.append({
            'player_id': player_id,
            'age': age,
            'country': country,
            'skill_level': skill_level,
            'daily_playtime_hours': round(daily_playtime, 2),
            'sessions_per_week': sessions_per_week,
            'is_paying_user': is_paying,
            'monthly_spend_usd': round(monthly_spend, 2),
            'win_rate': round(win_rate, 3),
            'friends_count': friends_count,
            'guild_member': guild_member,
            'days_since_last_login': days_since_last_login,
            'complaint_count': complaint_count,
            'will_churn': int(will_churn),
            'created_date': datetime.now() - timedelta(days=random.randint(30, 365))
        })
    
    return pd.DataFrame(players)

if __name__ == "__main__":
    df = generate_gaming_data(10000)
    df.to_csv('data/datasets/gaming_data.csv', index=False)
    print("Gaming dataset created successfully!")
    print(df.head())
    print(f"\nDataset shape: {df.shape}")
    print(f"Churn rate: {df['will_churn'].mean():.2%}")
