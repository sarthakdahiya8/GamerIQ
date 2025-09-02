# 🎮 GamerIQ – AI/ML Gaming Analytics Platform

## Introduction

**GamerIQ** is a machine learning-powered analytics platform designed for the gaming industry. It helps game companies and developers analyze player behavior, predict player churn, segment users, and optimize engagement through data-driven recommendations. The project includes a Flask REST API for interacting with ML models and a user-friendly Streamlit dashboard for visualizing insights in real time. GamerIQ is built to be modular, scalable, and easy to deploy.

---


## ✨ Features
- Churn prediction (XGBoost/RandomForest/LogReg comparison) with engineered engagement, monetization, social, and recency features
- Player segmentation using K-Means into 4 behavior clusters
- Engagement optimization suggestions with simulated uplift
- REST API endpoints (Flask) + Interactive Dashboard (Streamlit)
- Synthetic data generator to bootstrap development


## How to Run GamerIQ Locally

### 1. Clone the Repository

git clone https://github.com/sarthakdahiya8/GamerIQ.git

cd GamerIQ

### 2. Set up a Python Virtual Environment

python -m venv venv

For Windows:

venv\Scripts\activate

For macOS/Linux:

source venv/bin/activate

### 3. Install Project Dependencies

pip install -r requirements.txt

### 4. Generate Synthetic Gaming Data

python data/synthetic_game_data.py

*This will create a sample dataset (`data/datasets/gaming_data.csv`) used by the ML models.*

### 5. Train Machine Learning Models

python models/player_churn_prediction.py

*This will train churn prediction models and save the best model for API inference.*

### 6. Start the Flask API Server

python api/flask_api.py

*Test the API health endpoint in your browser:*

http://127.0.0.1:5000/health


### 7. Launch the Streamlit Dashboard (in a new terminal)

streamlit run api/dashboard.py

*This will open a local dashboard to visualize analytics and interact with ML models.*

---

## Notes
- Data (`.csv` files) and pickled models are auto-generated during steps 4 & 5.

## 🔌 API Endpoints
- GET `/health` → API health check
- POST `/api/player/churn-risk` → predict churn risk for a player
- POST `/api/player/engagement-boost` → simulate engagement uplift for an intervention
- POST `/api/player/recommendations` → personalized player recommendations
- GET `/api/analytics/dashboard` → aggregated analytics for dashboard

## 🧠 Tech Stack
- Python, Pandas, NumPy
- scikit-learn, XGBoost
- Flask (API), Streamlit (UI)
- Plotly/Seaborn/Matplotlib
- Gunicorn (production WSGI)

## 📈 Roadmap
- Time-series models (LSTM) for session sequences
- A/B testing harness for interventions
- DB integration (PostgreSQL/MongoDB)
- Auth + role-based dashboard
- CI checks and linting

## 👤 Author
Sarthak Dahiya  
B.Tech CSE, Bennett University (2022–2026)  
Interests: Gaming Analytics, ML, Full-Stack Development
