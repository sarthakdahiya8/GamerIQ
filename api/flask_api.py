from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.player_churn_prediction import ChurnPredictor
from models.engagement_optimizer import EngagementOptimizer

app = Flask(__name__)

# Global models
churn_predictor = ChurnPredictor()
engagement_optimizer = EngagementOptimizer()

# Load pre-trained models
try:
    churn_predictor.load_model('models/churn_prediction_model.pkl')
    print("Churn prediction model loaded successfully")
except:
    print("Churn model not found. Please train the model first.")

@app.route('/api/player/churn-risk', methods=['POST'])
def predict_churn_risk():
    """
    Player ka churn risk calculate karta hai
    """
    try:
        player_data = request.json
        
        # Convert to DataFrame
        df = pd.DataFrame([player_data])
        
        # Preprocess
        X, _ = churn_predictor.preprocess_data(df)
        
        # Predict
        result = churn_predictor.predict_churn_risk(X)
        
        return jsonify({
            'success': True,
            'data': result
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/player/engagement-boost', methods=['POST'])
def predict_engagement_boost():
    """
    Engagement interventions ka impact predict karta hai
    """
    try:
        data = request.json
        intervention = data.get('intervention_type')
        player_data = data.get('player_data')
        
        result = engagement_optimizer.predict_engagement_boost(intervention, player_data)
        
        return jsonify({
            'success': True,
            'data': result
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/player/recommendations', methods=['POST'])
def get_recommendations():
    """
    Personalized recommendations generate karta hai
    """
    try:
        data = request.json
        player_id = data.get('player_id')
        
        # Load player data (you'd get this from your database)
        df = pd.read_csv('data/datasets/gaming_data.csv')
        df = engagement_optimizer.segment_players(df)
        
        recommendations = engagement_optimizer.get_personalized_recommendations(player_id, df)
        
        return jsonify({
            'success': True,
            'data': recommendations
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/analytics/dashboard', methods=['GET'])
def get_dashboard_data():
    """
    Real-time analytics dashboard data
    """
    try:
        df = pd.read_csv('data/datasets/gaming_data.csv')
        
        # Key metrics
        total_players = len(df)
        active_players = len(df[df['days_since_last_login'] <= 7])
        paying_users = len(df[df['is_paying_user'] == 1])
        churn_rate = df['will_churn'].mean()
        
        # Segment analysis
        df = engagement_optimizer.segment_players(df)
        segment_distribution = df['player_segment'].value_counts().to_dict()
        
        dashboard_data = {
            'key_metrics': {
                'total_players': total_players,
                'active_players': active_players,
                'paying_users': paying_users,
                'churn_rate': round(churn_rate * 100, 2),
                'retention_rate': round((1 - churn_rate) * 100, 2)
            },
            'segment_distribution': segment_distribution,
            'avg_engagement_by_segment': df.groupby('player_segment')['daily_playtime_hours'].mean().round(2).to_dict()
        }
        
        return jsonify({
            'success': True,
            'data': dashboard_data
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/health', methods=['GET'])
def health_check():
    return {
        "status": "healthy",
        "message": "GamerIQ API is running!"
    }

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
