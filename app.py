import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from src.data_loader import ReservoirDataLoader, DEFAULT_DAM_NAME, DEFAULT_LATITUDE, DEFAULT_LONGITUDE, DEFAULT_SURFACE_AREA_KM2, DEFAULT_MAX_CAPACITY_MCM
from src.physics_penman import calculate_fao56_penman_monteith

# Page Configuration
st.set_page_config(
    page_title="AI Reservoir Evaporation & Water Level Predictor",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1b4f72;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #566573;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #ebf5fb 0%, #d4efdf 100%);
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 5px solid #2980b9;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1a5276;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #515a5a;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown("<div class='main-title'>🌊 AI-Based Reservoir Evaporation & Water Level Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Physics-Informed Machine Learning System for Dam Evaporation Estimation & Volumetric Loss Analysis</div>", unsafe_allow_html=True)

# Load Trained Models & Dataset
@st.cache_resource
def load_model_artifacts():
    model_path = "models/trained_models.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

@st.cache_data
def load_historical_dataset():
    data_path = "data/khadakwasla_evaporation_dataset.csv"
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        df['date'] = pd.to_datetime(df['date'])
        return df
    return None

artifacts = load_model_artifacts()
hist_df = load_historical_dataset()

# Sidebar Setup
st.sidebar.image("https://img.icons8.com/color/96/dam.png", width=70)
st.sidebar.title("🏞️ Reservoir & Location Settings")

preset_dams = {
    "Khadakwasla Dam (Pune, India)": {"lat": 18.4419, "lon": 73.7628, "area": 14.8, "capacity": 86.0},
    "Koyna Dam (Maharashtra, India)": {"lat": 17.3986, "lon": 73.7490, "area": 115.3, "capacity": 2797.0},
    "Tehri Dam (Uttarakhand, India)": {"lat": 30.3775, "lon": 78.4800, "area": 52.0, "capacity": 3540.0},
    "Custom Location / Dam": {"lat": DEFAULT_LATITUDE, "lon": DEFAULT_LONGITUDE, "area": 10.0, "capacity": 50.0}
}

selected_dam = st.sidebar.selectbox("Select Target Reservoir:", list(preset_dams.keys()))

if selected_dam == "Custom Location / Dam":
    lat = st.sidebar.number_input("Latitude (°N):", value=DEFAULT_LATITUDE, format="%.4f")
    lon = st.sidebar.number_input("Longitude (°E):", value=DEFAULT_LONGITUDE, format="%.4f")
    surface_area = st.sidebar.number_input("Reservoir Surface Area (km²):", value=10.0, min_value=0.1, step=1.0)
    capacity_mcm = st.sidebar.number_input("Max Capacity (Million m³):", value=50.0, min_value=1.0, step=5.0)
else:
    info = preset_dams[selected_dam]
    lat, lon = info["lat"], info["lon"]
    surface_area = info["area"]
    capacity_mcm = info["capacity"]
    st.sidebar.info(f"📍 **Coordinates:** {lat}°N, {lon}°E\n📐 **Surface Area:** {surface_area} km²\n💧 **Capacity:** {capacity_mcm} MCM")

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Active ML Models")
selected_model_name = st.sidebar.selectbox(
    "Primary Predictor Model:", 
    ["FAO-56 Penman Monteith (Physics Baseline)", "Multi-Layer Perceptron (ANN)", "Support Vector Regressor (SVR)", "Gradient Boosting Regressor", "Extra Trees Regressor", "Random Forest Regressor", "Ridge Regression"]
)

# Tabs Navigation
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌐 Live Forecast & Evaporation Predictor", 
    "📊 Exploratory Data Analysis (EDA)", 
    "🏆 Model Comparison Leaderboard", 
    "💧 Volumetric Loss & Water Level", 
    "🧪 What-If Climate Simulator"
])

# ==================== TAB 1: LIVE FORECAST & PREDICTOR ====================
with tab1:
    st.subheader("🗓️ Real-Time 7-Day Weather & Reservoir Evaporation Forecast")
    
    loader = ReservoirDataLoader(lat=lat, lon=lon, dam_name=selected_dam)
    with st.spinner("Fetching live weather forecast from satellite API..."):
        try:
            live_df = loader.fetch_live_forecast(days=7)
            live_df = calculate_fao56_penman_monteith(live_df)
            
            # Predict using selected ML model
            live_df['doy'] = live_df['date'].dt.dayofyear
            features = ['temp_max_c', 'temp_min_c', 'temp_mean_c', 'humidity_pct', 'wind_speed_2m_ms', 'solar_radiation_mj', 'sunshine_hours', 'pressure_hpa', 'vpd_kpa', 'net_radiation_mj', 'doy']
            
            if selected_model_name == "FAO-56 Penman Monteith (Physics Baseline)":
                live_df['predicted_evaporation_mm'] = (live_df['penman_evaporation_mm'] / 0.75).round(2)
            elif artifacts and selected_model_name in artifacts['models']:
                model_obj = artifacts['models'][selected_model_name]
                live_df['predicted_evaporation_mm'] = model_obj.predict(live_df[features]).round(2)
            else:
                live_df['predicted_evaporation_mm'] = live_df['evaporation_pan_mm']
                
            # Volume loss calculations
            # 1 mm over 1 km² = 1,000,000 Liters = 1,000 m³ = 0.001 MCM
            live_df['volume_loss_liters'] = (live_df['predicted_evaporation_mm'] * surface_area * 1e6).round(0)
            live_df['volume_loss_mcm'] = (live_df['predicted_evaporation_mm'] * surface_area * 0.001).round(4)
            
            # Key Summary Metric Cards
            col1, col2, col3, col4 = st.columns(4)
            total_7day_evap_mm = live_df['predicted_evaporation_mm'].sum()
            total_7day_loss_liters = live_df['volume_loss_liters'].sum()
            total_7day_loss_mcm = live_df['volume_loss_mcm'].sum()
            pune_daily_demand_liters = 1.4e9 # Pune city daily water supply requirement ~1,400 MLD
            days_of_water_lost = total_7day_loss_liters / pune_daily_demand_liters
            
            with col1:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>7-Day Total Evaporation Depth</div>
                    <div class='metric-value'>{total_7day_evap_mm:.2f} mm</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>Total Volume Lost (Liters)</div>
                    <div class='metric-value'>{total_7day_loss_liters/1e6:.2f} Million L</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>Total Volume Lost (MCM)</div>
                    <div class='metric-value'>{total_7day_loss_mcm:.3f} MCM</div>
                </div>
                """, unsafe_allow_html=True)
            with col4:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>Pune City Water Supply Equiv.</div>
                    <div class='metric-value'>{days_of_water_lost:.1f} Days</div>
                </div>
                """, unsafe_allow_html=True)
                
            # Forecast Plot
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=live_df['date'].dt.strftime('%a %d %b'),
                y=live_df['predicted_evaporation_mm'],
                name=f'Predicted Evaporation ({selected_model_name})',
                marker_color='#2980b9'
            ))
            fig.add_trace(go.Scatter(
                x=live_df['date'].dt.strftime('%a %d %b'),
                y=live_df['temp_mean_c'],
                name='Mean Temperature (°C)',
                yaxis='y2',
                line=dict(color='#e74c3c', width=3)
            ))
            
            fig.update_layout(
                title="7-Day Predicted Evaporation Rate vs Mean Temperature",
                xaxis=dict(title="Date"),
                yaxis=dict(title="Daily Evaporation Depth (mm/day)", side='left'),
                yaxis2=dict(title="Temperature (°C)", overlaying='y', side='right'),
                legend=dict(x=0.01, y=1.15, orientation='h'),
                template="plotly_white",
                height=450
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.write("### 📋 7-Day Live Meteorological & Evaporation Table")
            display_cols = ['date', 'temp_mean_c', 'humidity_pct', 'wind_speed_2m_ms', 'solar_radiation_mj', 'predicted_evaporation_mm', 'volume_loss_liters', 'volume_loss_mcm']
            st.dataframe(live_df[display_cols].style.format({
                'temp_mean_c': '{:.1f} °C',
                'humidity_pct': '{:.1f} %',
                'wind_speed_2m_ms': '{:.2f} m/s',
                'solar_radiation_mj': '{:.2f} MJ/m²',
                'predicted_evaporation_mm': '{:.2f} mm',
                'volume_loss_liters': '{:,.0f} L',
                'volume_loss_mcm': '{:.4f} MCM'
            }), use_container_width=True)
            
        except Exception as e:
            st.error(f"Error fetching live forecast data: {e}")

# ==================== TAB 2: EXPLORATORY DATA ANALYSIS ====================
with tab2:
    st.subheader("📊 Exploratory Data Analysis & Diagnostic Insights")
    if hist_df is not None:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### 1. Weather Feature Correlation Matrix")
            if os.path.exists("data/eda_plots/correlation_matrix.png"):
                st.image("data/eda_plots/correlation_matrix.png", use_container_width=True)
        with col_b:
            st.markdown("#### 2. Evaporation Drivers (Temperature, Solar, Humidity, Wind)")
            if os.path.exists("data/eda_plots/evaporation_vs_weather.png"):
                st.image("data/eda_plots/evaporation_vs_weather.png", use_container_width=True)
                
        st.markdown("#### 3. Temporal & Seasonal Evaporation Trends (2021-2024)")
        if os.path.exists("data/eda_plots/seasonal_evaporation_trend.png"):
            st.image("data/eda_plots/seasonal_evaporation_trend.png", use_container_width=True)
    else:
        st.warning("Historical EDA dataset not found. Run pipeline first.")

# ==================== TAB 3: MODEL LEADERBOARD ====================
with tab3:
    st.subheader("🏆 Multi-Model Performance Leaderboard & Analysis")
    if artifacts is not None and 'leaderboard' in artifacts:
        leaderboard = artifacts['leaderboard']
        st.dataframe(leaderboard.style.highlight_max(subset=['R2 Score', 'NSE'], color='#d4efdf').highlight_min(subset=['RMSE (mm/day)', 'MAE (mm/day)', 'MAPE (%)'], color='#d4efdf'), use_container_width=True)
        
        # Leaderboard Bar Chart
        fig_lead = px.bar(
            leaderboard, x='Model', y='R2 Score', color='R2 Score',
            title="Model Comparison by R² Score (Physics vs ML/DL Models)",
            color_continuous_scale="Viridis", text_auto='.4f'
        )
        fig_lead.update_layout(template="plotly_white", height=400)
        st.plotly_chart(fig_lead, use_container_width=True)
        
        # Test Set Prediction Comparison
        test_df = artifacts['test_predictions']
        st.markdown("#### 📈 Model Predictions vs Actual Hardware Evaporation Pan (Test Set)")
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Scatter(x=test_df['date'], y=test_df['Actual_Pan_Evaporation'], name='Actual Pan Sensor', line=dict(color='black', width=2)))
        
        for m_name in leaderboard['Model']:
            if m_name in test_df.columns:
                fig_comp.add_trace(go.Scatter(x=test_df['date'], y=test_df[m_name], name=m_name, line=dict(width=1.5), visible='legendonly' if 'Ridge' in m_name else True))
                
        fig_comp.update_layout(title="Time Series Comparison of Test Set Predictions", template="plotly_white", height=450)
        st.plotly_chart(fig_comp, use_container_width=True)

# ==================== TAB 4: VOLUMETRIC LOSS & WATER LEVEL ====================
with tab4:
    st.subheader("💧 Volumetric Water Loss & Reservoir Level Calculator")
    
    col1, col2 = st.columns(2)
    with col1:
        calc_area = st.number_input("Current Reservoir Surface Area (km²):", value=float(surface_area), min_value=0.1)
        calc_evap_rate = st.number_input("Daily Evaporation Rate (mm/day):", value=6.5, min_value=0.1, max_value=25.0)
        calc_days = st.slider("Forecast Duration (Days):", min_value=1, max_value=90, value=30)
        
    with col2:
        calc_current_storage = st.number_input("Current Dam Storage (Million m³ / MCM):", value=float(capacity_mcm * 0.75), min_value=0.1)
        calc_max_storage = float(capacity_mcm)
        
    # Volumetric calculations
    total_depth_mm = calc_evap_rate * calc_days
    total_loss_m3 = (calc_evap_rate * 0.001) * (calc_area * 1e6) * calc_days
    total_loss_liters = total_loss_m3 * 1000
    total_loss_mcm = total_loss_m3 / 1e6
    
    remaining_storage = max(0.0, calc_current_storage - total_loss_mcm)
    pct_storage_lost = (total_loss_mcm / calc_current_storage) * 100 if calc_current_storage > 0 else 0
    water_level_drop_cm = (total_depth_mm / 10.0)
    
    st.markdown("---")
    st.markdown(f"### 📊 Projected Volumetric Loss over {calc_days} Days")
    
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Total Evaporation Depth", f"{total_depth_mm:.1f} mm", f"Water Drop: {water_level_drop_cm:.1f} cm")
    mc2.metric("Total Volume Lost", f"{total_loss_liters/1e9:.3f} Billion L", f"{total_loss_mcm:.2f} MCM")
    mc3.metric("Storage Depletion", f"{pct_storage_lost:.2f} %", f"Remaining: {remaining_storage:.2f} MCM")
    mc4.metric("Pune City Supply Equiv.", f"{total_loss_liters/1.4e9:.1f} Days of Water")

    # Gauge Chart
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=remaining_storage,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Reservoir Available Storage (MCM)"},
        delta={'reference': calc_current_storage, 'increasing': {'color': "green"}},
        gauge={
            'axis': {'range': [0, calc_max_storage]},
            'bar': {'color': "#2980b9"},
            'steps': [
                {'range': [0, calc_max_storage * 0.25], 'color': "#fadbd8"},
                {'range': [calc_max_storage * 0.25, calc_max_storage * 0.75], 'color': "#fdebd0"},
                {'range': [calc_max_storage * 0.75, calc_max_storage], 'color': "#d4efdf"}
            ]
        }
    ))
    fig_gauge.update_layout(height=350, template="plotly_white")
    st.plotly_chart(fig_gauge, use_container_width=True)

# ==================== TAB 5: WHAT-IF CLIMATE SIMULATOR ====================
with tab5:
    st.subheader("🧪 'What-If' Climate Change & Pan Calibration Simulator")
    
    st.write("Simulate how temperature rise, solar radiation increase, or wind speed changes impact daily reservoir evaporation loss.")
    
    sc_col1, sc_col2, sc_col3 = st.columns(3)
    with sc_col1:
        temp_delta = st.slider("Temperature Rise (°C):", 0.0, 5.0, 1.5, step=0.5)
    with sc_col2:
        humidity_delta = st.slider("Relative Humidity Shift (%):", -20.0, 20.0, -5.0, step=1.0)
    with sc_col3:
        wind_delta = st.slider("Wind Speed Change (m/s):", -2.0, 3.0, 0.5, step=0.2)
        
    if hist_df is not None:
        sim_df = hist_df.copy()
        sim_df['temp_mean_c'] += temp_delta
        sim_df['temp_max_c'] += temp_delta
        sim_df['temp_min_c'] += temp_delta
        sim_df['humidity_pct'] = np.clip(sim_df['humidity_pct'] + humidity_delta, 10, 100)
        sim_df['wind_speed_2m_ms'] = np.maximum(0.2, sim_df['wind_speed_2m_ms'] + wind_delta)
        
        sim_df = calculate_fao56_penman_monteith(sim_df)
        
        base_avg_evap = hist_df['evaporation_pan_mm'].mean()
        sim_avg_evap = sim_df['evaporation_pan_mm'].mean()
        pct_increase = ((sim_avg_evap - base_avg_evap) / base_avg_evap) * 100
        
        st.info(f"🔥 **Baseline Avg Evaporation:** {base_avg_evap:.2f} mm/day | 🧪 **Simulated Avg Evaporation:** {sim_avg_evap:.2f} mm/day | 📈 **Net Increase:** +{pct_increase:.1f}%")
        
        fig_sim = go.Figure()
        fig_sim.add_trace(go.Scatter(x=hist_df['date'], y=hist_df['evaporation_pan_mm'], name="Historical Baseline Evaporation", line=dict(color="#2980b9", width=1.5)))
        fig_sim.add_trace(go.Scatter(x=sim_df['date'], y=sim_df['evaporation_pan_mm'], name="Simulated Evaporation (+Climate Shift)", line=dict(color="#e74c3c", width=1.5)))
        fig_sim.update_layout(title="Climate Shift Simulation vs Historical Evaporation Baseline", template="plotly_white", height=400)
        st.plotly_chart(fig_sim, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #7f8c8d;'>Semester Main Project: AI-Based Reservoir Evaporation & Water Level Prediction System | Khadakwasla Dam Baseline & Global Location Platform</div>", unsafe_allow_html=True)
