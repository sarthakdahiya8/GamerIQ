import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# Path setup
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.player_churn_prediction import ChurnPredictor
from models.engagement_optimizer import EngagementOptimizer

# Page config
st.set_page_config(
    page_title="GamerIQ - AI Gaming Analytics",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    text-align: center;
    background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 2rem;
}
.metric-card {
    background: #1E1E1E;
    padding: 1rem;
    border-radius: 10px;
    border-left: 4px solid #4ECDC4;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Data load karta hai"""
    df = pd.read_csv('data/datasets/gaming_data.csv')
    return df

def main():
    st.markdown('<h1 class="main-header">🎮 GamerIQ Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose Analysis Type",
        ["Dashboard Overview", "Churn Prediction", "Player Segmentation", "Engagement Optimization", "Real-time Analytics"]
    )
    
    # Load data
    df = load_data()
    
    if page == "Dashboard Overview":
        show_dashboard_overview(df)
    elif page == "Churn Prediction":
        show_churn_prediction(df)
    elif page == "Player Segmentation":
        show_player_segmentation(df)
    elif page == "Engagement Optimization":
        show_engagement_optimization(df)
    elif page == "Real-time Analytics":
        show_realtime_analytics(df)

def show_dashboard_overview(df):
    """Main dashboard overview"""
    st.header("📊 Gaming Analytics Overview")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Players",
            value=f"{len(df):,}",
            delta="↗️ +12% vs last month"
        )
    
    with col2:
        active_players = len(df[df['days_since_last_login'] <= 7])
        st.metric(
            label="Active Players (7d)",
            value=f"{active_players:,}",
            delta=f"{(active_players/len(df)*100):.1f}% of total"
        )
    
    with col3:
        paying_users = len(df[df['is_paying_user'] == 1])
        st.metric(
            label="Paying Users",
            value=f"{paying_users:,}",
            delta=f"{(paying_users/len(df)*100):.1f}% conversion"
        )
    
    with col4:
        churn_rate = df['will_churn'].mean() * 100
        st.metric(
            label="Churn Rate",
            value=f"{churn_rate:.1f}%",
            delta="↘️ -2.3% vs last month"
        )
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Player distribution by country
        fig_country = px.pie(
            df['country'].value_counts().reset_index(),
            values='count',
            names='country',
            title="Player Distribution by Country"
        )
        st.plotly_chart(fig_country, use_container_width=True)
    
    with col2:
        # Engagement vs Churn
        fig_engagement = px.scatter(
            df,
            x='daily_playtime_hours',
            y='monthly_spend_usd',
            color='will_churn',
            title="Engagement vs Monetization",
            labels={'will_churn': 'Will Churn'}
        )
        st.plotly_chart(fig_engagement, use_container_width=True)

def show_churn_prediction(df):
    """Churn prediction interface"""
    st.header("🔮 Player Churn Prediction")
    
    st.subheader("Individual Player Risk Assessment")
    
    # Input form
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", 16, 50, 25)
        daily_playtime = st.slider("Daily Playtime (hours)", 0.5, 12.0, 2.5)
        sessions_per_week = st.slider("Sessions per Week", 1, 20, 8)
        win_rate = st.slider("Win Rate", 0.0, 1.0, 0.5)
    
    with col2:
        monthly_spend = st.slider("Monthly Spend ($)", 0, 200, 25)
        friends_count = st.slider("Friends Count", 0, 100, 15)
        days_since_login = st.slider("Days Since Last Login", 0, 30, 3)
        complaint_count = st.slider("Complaint Count", 0, 10, 1)
    
    # Country and skill level
    country = st.selectbox("Country", ['India', 'USA', 'UK', 'Brazil', 'Germany'])
    skill_level = st.selectbox("Skill Level", ['Beginner', 'Intermediate', 'Advanced', 'Pro'])
    
    if st.button("Predict Churn Risk", type="primary"):
        # Create player data
        player_data = {
            'age': age,
            'country': country,
            'skill_level': skill_level,
            'daily_playtime_hours': daily_playtime,
            'sessions_per_week': sessions_per_week,
            'is_paying_user': 1 if monthly_spend > 0 else 0,
            'monthly_spend_usd': monthly_spend,
            'win_rate': win_rate,
            'friends_count': friends_count,
            'guild_member': 1,  # Assume guild member
            'days_since_last_login': days_since_login,
            'complaint_count': complaint_count,
            'will_churn': 0  # Placeholder
        }
        
        # Mock prediction (replace with actual model)
        churn_prob = min(0.8, (
            (age > 30) * 0.1 +
            (daily_playtime < 1) * 0.3 +
            (monthly_spend == 0) * 0.2 +
            (days_since_login > 7) * 0.4 +
            (complaint_count > 2) * 0.3 +
            (friends_count < 5) * 0.1
        ))
        
        # Results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Churn Probability", f"{churn_prob:.1%}")
        
        with col2:
            risk_level = "Low" if churn_prob < 0.3 else "Medium" if churn_prob < 0.6 else "High"
            color = "🟢" if risk_level == "Low" else "🟡" if risk_level == "Medium" else "🔴"
            st.metric("Risk Level", f"{color} {risk_level}")
        
        with col3:
            retention_score = (1 - churn_prob) * 100
            st.metric("Retention Score", f"{retention_score:.0f}/100")
        
        # Recommendations
        st.subheader("Retention Strategy")
        if churn_prob < 0.3:
            st.success("✅ Player is well-engaged. Continue current strategies and regular content updates.")
        elif churn_prob < 0.6:
            st.warning("⚠️ Moderate risk. Consider personalized offers, social features, and engagement campaigns.")
        else:
            st.error("🚨 High churn risk! Immediate intervention needed: exclusive rewards, personal support, special events.")

def show_player_segmentation(df):
    """Player segmentation analysis"""
    st.header("👥 Player Segmentation Analysis")
    
    # Initialize optimizer
    optimizer = EngagementOptimizer()
    df_segmented = optimizer.segment_players(df.copy())
    
    # Segment overview
    segment_names = {0: 'Casual Players', 1: 'Hardcore Gamers', 2: 'Social Players', 3: 'Competitive Players'}
    df_segmented['segment_name'] = df_segmented['player_segment'].map(segment_names)
    
    # Segment distribution
    fig_segments = px.pie(
        df_segmented['segment_name'].value_counts().reset_index(),
        values='count',
        names='segment_name',
        title="Player Segment Distribution"
    )
    st.plotly_chart(fig_segments, use_container_width=True)
    
    # Segment characteristics
    st.subheader("Segment Characteristics")
    
    segment_stats = df_segmented.groupby('segment_name').agg({
        'daily_playtime_hours': 'mean',
        'sessions_per_week': 'mean',
        'monthly_spend_usd': 'mean',
        'win_rate': 'mean',
        'friends_count': 'mean',
        'will_churn': 'mean'
    }).round(2)
    
    st.dataframe(segment_stats, use_container_width=True)
    
    # Segment comparison charts
    col1, col2 = st.columns(2)
    
    with col1:
        fig_playtime = px.box(
            df_segmented,
            x='segment_name',
            y='daily_playtime_hours',
            title="Daily Playtime by Segment"
        )
        st.plotly_chart(fig_playtime, use_container_width=True)
    
    with col2:
        fig_spend = px.box(
            df_segmented,
            x='segment_name',
            y='monthly_spend_usd',
            title="Monthly Spend by Segment"
        )
        st.plotly_chart(fig_spend, use_container_width=True)

def show_engagement_optimization(df):
    """Engagement optimization tools"""
    st.header("🚀 Engagement Optimization")
    
    st.subheader("Intervention Impact Simulator")
    
    # Player selection
    player_id = st.selectbox("Select Player", df['player_id'].head(20))
    player_data = df[df['player_id'] == player_id].iloc[0]
    
    # Show player info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Playtime", f"{player_data['daily_playtime_hours']:.1f}h/day")
    with col2:
        st.metric("Sessions/Week", f"{player_data['sessions_per_week']}")
    with col3:
        st.metric("Monthly Spend", f"${player_data['monthly_spend_usd']:.0f}")
    
    # Intervention selection
    intervention = st.selectbox(
        "Select Intervention",
        ['daily_bonus', 'social_event', 'competitive_tournament', 'personalized_content', 'vip_rewards']
    )
    
    if st.button("Simulate Impact", type="primary"):
        # Mock impact calculation
        interventions = {
            'daily_bonus': {'playtime_boost': 0.15, 'description': 'Daily login bonuses'},
            'social_event': {'playtime_boost': 0.08, 'description': 'Community events'},
            'competitive_tournament': {'playtime_boost': 0.25, 'description': 'Competitive tournaments'},
            'personalized_content': {'playtime_boost': 0.12, 'description': 'Personalized game content'},
            'vip_rewards': {'playtime_boost': 0.10, 'description': 'VIP reward program'}
        }
        
        boost = interventions[intervention]['playtime_boost']
        current_engagement = player_data['daily_playtime_hours']
        predicted_new_engagement = current_engagement * (1 + boost)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Engagement", f"{current_engagement:.1f}h")
        with col2:
            st.metric("Predicted Boost", f"+{boost*100:.0f}%")
        with col3:
            st.metric("New Engagement", f"{predicted_new_engagement:.1f}h")
        
        st.success(f"✅ {interventions[intervention]['description']} could increase daily playtime by {boost*100:.0f}%")

def show_realtime_analytics(df):
    """Real-time analytics simulation"""
    st.header("⚡ Real-time Analytics")
    
    # Auto-refresh toggle
    auto_refresh = st.checkbox("Auto-refresh (simulated)")
    
    if auto_refresh:
        # Simulate real-time data
        import time
        placeholder = st.empty()
        
        for i in range(10):
            with placeholder.container():
                # Simulated real-time metrics
                current_players = np.random.randint(800, 1200)
                new_signups = np.random.randint(10, 50)
                revenue_today = np.random.uniform(5000, 15000)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Players Online", current_players, f"+{np.random.randint(-50, 100)}")
                with col2:
                    st.metric("New Signups Today", new_signups, f"+{np.random.randint(0, 10)}")
                with col3:
                    st.metric("Revenue Today", f"${revenue_today:.0f}", f"+{np.random.uniform(-500, 1000):.0f}")
                
                # Real-time chart
                time_data = pd.DataFrame({
                    'hour': range(24),
                    'active_players': np.random.randint(200, 1000, 24)
                })
                
                fig_realtime = px.line(
                    time_data,
                    x='hour',
                    y='active_players',
                    title="Active Players (Last 24 Hours)"
                )
                st.plotly_chart(fig_realtime, use_container_width=True)
            
            time.sleep(2)  # Update every 2 seconds
    else:
        st.info("Enable auto-refresh to see simulated real-time data")

if __name__ == "__main__":
    main()
