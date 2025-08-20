import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class ChurnPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_importance = None
        
    def preprocess_data(self, df):
        """
        Gaming data ko ML ready banata hai
        """
        df = df.copy()
        
        # Categorical encoding
        categorical_cols = ['country', 'skill_level']
        for col in categorical_cols:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                df[col] = self.encoders[col].fit_transform(df[col])
            else:
                df[col] = self.encoders[col].transform(df[col])
        
        # Feature engineering
        df['engagement_score'] = (
            df['daily_playtime_hours'] * 0.3 +
            df['sessions_per_week'] * 0.2 +
            df['win_rate'] * 50 * 0.3 +
            (df['friends_count'] / 50) * 0.2
        )
        
        df['monetization_score'] = df['monthly_spend_usd'] / 100
        df['social_score'] = (df['friends_count'] / 100) + df['guild_member']
        df['activity_recency'] = 1 / (1 + df['days_since_last_login'])
        
        # Feature selection
        features = [
            'age', 'daily_playtime_hours', 'sessions_per_week', 
            'is_paying_user', 'monthly_spend_usd', 'win_rate',
            'friends_count', 'guild_member', 'days_since_last_login',
            'complaint_count', 'country', 'skill_level',
            'engagement_score', 'monetization_score', 'social_score', 'activity_recency'
        ]
        
        X = df[features]
        y = df['will_churn']
        
        return X, y
    
    def train_models(self, X, y):
        """
        Multiple models train karta hai aur best select karta hai
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale numerical features
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        self.scalers['standard'] = StandardScaler()
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        X_train_scaled[numerical_cols] = self.scalers['standard'].fit_transform(X_train[numerical_cols])
        X_test_scaled[numerical_cols] = self.scalers['standard'].transform(X_test[numerical_cols])
        
        # Model definitions
        models_config = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'xgboost': xgb.XGBClassifier(random_state=42),
            'logistic_regression': LogisticRegression(random_state=42)
        }
        
        results = {}
        
        for name, model in models_config.items():
            print(f"\nTraining {name}...")
            
            if name == 'logistic_regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            results[name] = {
                'model': model,
                'auc_score': auc_score,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"{name} AUC Score: {auc_score:.4f}")
            print(classification_report(y_test, y_pred))
        
        # Best model select karo
        best_model_name = max(results.keys(), key=lambda x: results[x]['auc_score'])
        self.models['best'] = results[best_model_name]['model']
        self.models['best_name'] = best_model_name
        
        print(f"\nBest Model: {best_model_name} (AUC: {results[best_model_name]['auc_score']:.4f})")
        
        # Feature importance
        if hasattr(self.models['best'], 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.models['best'].feature_importances_
            }).sort_values('importance', ascending=False)
        
        return results, X_test, y_test
    
    def predict_churn_risk(self, player_data):
        """
        Real-time churn prediction
        """
        if self.models['best_name'] == 'logistic_regression':
            numerical_cols = player_data.select_dtypes(include=[np.number]).columns
            player_data_scaled = player_data.copy()
            player_data_scaled[numerical_cols] = self.scalers['standard'].transform(player_data[numerical_cols])
            churn_prob = self.models['best'].predict_proba(player_data_scaled)[:, 1]
        else:
            churn_prob = self.models['best'].predict_proba(player_data)[:, 1]
        
        risk_level = ['Low', 'Medium', 'High'][min(2, int(churn_prob[0] * 3))]
        
        return {
            'churn_probability': churn_prob[0],
            'risk_level': risk_level,
            'recommendation': self.get_retention_strategy(churn_prob[0])
        }
    
    def get_retention_strategy(self, churn_prob):
        """
        Churn probability ke basis pe retention strategy suggest karta hai
        """
        if churn_prob < 0.3:
            return "Player engaged hai - regular content updates aur community events continue karo"
        elif churn_prob < 0.6:
            return "Moderate risk - personalized offers aur social features promote karo"
        else:
            return "High risk - immediate intervention needed: exclusive rewards, personal support"
    
    def save_model(self, filepath):
        """Model save karta hai"""
        joblib.dump({
            'models': self.models,
            'scalers': self.scalers,
            'encoders': self.encoders,
            'feature_importance': self.feature_importance
        }, filepath)
    
    def load_model(self, filepath):
        """Model load karta hai"""
        data = joblib.load(filepath)
        self.models = data['models']
        self.scalers = data['scalers']
        self.encoders = data['encoders']
        self.feature_importance = data['feature_importance']

if __name__ == "__main__":
    # Data load karo
    df = pd.read_csv('../data/datasets/gaming_data.csv')
    
    # Model initialize aur train karo
    predictor = ChurnPredictor()
    X, y = predictor.preprocess_data(df)
    results, X_test, y_test = predictor.train_models(X, y)
    
    # Model save karo
    predictor.save_model('churn_prediction_model.pkl')
    
    # Feature importance plot
    if predictor.feature_importance is not None:
        plt.figure(figsize=(10, 8))
        sns.barplot(data=predictor.feature_importance.head(10), y='feature', x='importance')
        plt.title('Top 10 Features for Churn Prediction')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    print("\nChurn Prediction Model trained successfully!")
