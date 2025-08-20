import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

class EngagementOptimizer:
    def __init__(self):
        self.player_segments = None
        self.engagement_model = None
        self.optimal_strategies = {}
    
    def segment_players(self, df):
        """
        Players ko behavior ke basis pe segment karta hai
        """
        # Engagement features
        features = [
            'daily_playtime_hours', 'sessions_per_week', 'win_rate',
            'monthly_spend_usd', 'friends_count'
        ]
        
        # Normalize karo
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df[features])
        
        # K-means clustering
        kmeans = KMeans(n_clusters=4, random_state=42)
        df['player_segment'] = kmeans.fit_predict(scaled_features)
        
        # Segment analysis
        segment_analysis = df.groupby('player_segment').agg({
            'daily_playtime_hours': 'mean',
            'sessions_per_week': 'mean',
            'win_rate': 'mean',
            'monthly_spend_usd': 'mean',
            'friends_count': 'mean',
            'will_churn': 'mean'
        }).round(2)
        
        # Segment names assign karo
        segment_names = {
            0: 'Casual Players',
            1: 'Hardcore Gamers', 
            2: 'Social Players',
            3: 'Competitive Players'
        }
        
        self.player_segments = {
            'model': kmeans,
            'scaler': scaler,
            'analysis': segment_analysis,
            'names': segment_names,
            'features': features
        }
        
        return df
    
    def predict_engagement_boost(self, intervention_type, player_data):
        """
        Different interventions ka engagement pe impact predict karta hai
        """
        interventions = {
            'daily_bonus': {'playtime_boost': 0.15, 'session_boost': 0.1},
            'social_event': {'playtime_boost': 0.08, 'friends_boost': 0.2},
            'competitive_tournament': {'playtime_boost': 0.25, 'skill_boost': 0.05},
            'personalized_content': {'playtime_boost': 0.12, 'retention_boost': 0.3},
            'vip_rewards': {'spend_boost': 0.4, 'playtime_boost': 0.1}
        }
        
        if intervention_type not in interventions:
            return "Invalid intervention type"
        
        boost = interventions[intervention_type]
        current_engagement = (
            player_data['daily_playtime_hours'] * 0.4 +
            player_data['sessions_per_week'] * 0.3 +
            player_data['win_rate'] * 30 * 0.3
        )
        
        predicted_boost = sum(boost.values()) * current_engagement
        
        return {
            'current_engagement_score': current_engagement,
            'predicted_boost': predicted_boost,
            'new_engagement_score': current_engagement + predicted_boost,
            'percentage_improvement': (predicted_boost / current_engagement) * 100
        }
    
    def get_personalized_recommendations(self, player_id, df):
        """
        Individual player ke liye personalized recommendations
        """
        player = df[df['player_id'] == player_id].iloc[0]
        segment = player['player_segment']
        segment_name = self.player_segments['names'][segment]
        
        recommendations = []
        
        # Segment-based recommendations
        if segment_name == 'Casual Players':
            recommendations = [
                "Daily login bonuses to encourage regular play",
                "Simple achievement systems",
                "Tutorial improvements for better onboarding",
                "Weekend special events"
            ]
        elif segment_name == 'Hardcore Gamers':
            recommendations = [
                "Advanced difficulty modes",
                "Exclusive hardcore challenges",
                "Early access to new content",
                "Leaderboard competitions"
            ]
        elif segment_name == 'Social Players':
            recommendations = [
                "Guild system enhancements",
                "Multiplayer events and tournaments",
                "Social sharing features",
                "Friend referral rewards"
            ]
        else:  # Competitive Players
            recommendations = [
                "Ranked matchmaking improvements",
                "Esports tournament integration",
                "Advanced statistics and analytics",
                "Skill-based rewards system"
            ]
        
        # Individual player analysis
        individual_recommendations = []
        
        if player['daily_playtime_hours'] < 1:
            individual_recommendations.append("Short session rewards to accommodate limited playtime")
        
        if player['friends_count'] < 5:
            individual_recommendations.append("Social features promotion and friend-finding tools")
        
        if player['monthly_spend_usd'] == 0:
            individual_recommendations.append("Entry-level purchase incentives and value packs")
        
        if player['win_rate'] < 0.3:
            individual_recommendations.append("Skill improvement guides and practice modes")
        
        return {
            'player_segment': segment_name,
            'segment_recommendations': recommendations,
            'individual_recommendations': individual_recommendations,
            'priority_actions': individual_recommendations[:2]  # Top 2 priority actions
        }

if __name__ == "__main__":
    # Test the engagement optimizer
    df = pd.read_csv('../data/datasets/gaming_data.csv')
    
    optimizer = EngagementOptimizer()
    df = optimizer.segment_players(df)
    
    print("Player Segmentation Analysis:")
    print(optimizer.player_segments['analysis'])
    
    # Test recommendations
    sample_player = df.iloc[0]
    recommendations = optimizer.get_personalized_recommendations(sample_player['player_id'], df)
    print(f"\nRecommendations for {sample_player['player_id']}:")
    print(f"Segment: {recommendations['player_segment']}")
    print("Individual recommendations:", recommendations['individual_recommendations'])
