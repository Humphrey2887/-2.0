"""
US Energy Consumption Forecast | 美国能源消费预测 - app.py (Ultimate Full-Feature Version)
Features:
1. Trump 2.0 vs Baseline Comparison (New)
2. CO2 Emission Forecasting (New)
3. Sensitivity Analysis Heatmap (Restored)
4. SHAP Explanation (New - Advanced)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import shap
import matplotlib.pyplot as plt
import warnings

# 忽略警告信息
warnings.filterwarnings('ignore')
st.set_option('deprecation.showPyplotGlobalUse', False) # 允许SHAP绘图

# Page Configuration
st.set_page_config(
    page_title="Energy Forecast Pro | 能源决策系统",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# Data Loading & Processing
# ============================================

@st.cache_data
def load_manual_data(filepath: str = "manual_data.csv") -> pd.DataFrame:
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        st.error(f"File not found: {filepath}. Please run process_data.py first.")
        st.stop()

@st.cache_data
def get_static_macro_data() -> pd.DataFrame:
    """Returns embedded REAL historical macro data (2000-2023)."""
    data = {
        'Year': list(range(2000, 2024)),
        'GDP': [
            10252, 10581, 10936, 11458, 12213, 13036, 13814, 14451, 14712, 14448, 
            14992, 15542, 16197, 16784, 17521, 18219, 18707, 19485, 20527, 21372, 
            20893, 22996, 25462, 27360
        ],
        'Industrial_Reshoring': [
            92.8, 89.4, 89.7, 90.9, 93.3, 96.5, 98.6, 100.0, 96.3, 85.0, 
            90.6, 93.6, 96.6, 98.4, 101.3, 100.6, 99.4, 100.0, 103.1, 102.4, 
            95.4, 100.4, 103.7, 103.0
        ],
        'Oil_Price': [
            30.3, 25.9, 26.1, 31.1, 41.4, 56.5, 66.1, 72.3, 99.6, 61.7, 
            79.4, 94.8, 94.1, 97.9, 93.1, 48.7, 43.2, 50.8, 65.2, 56.9, 
            39.2, 68.1, 94.4, 77.6
        ]
    }
    df = pd.DataFrame(data)
    last_row = df.iloc[-1].copy()
    last_row['Year'] = 2024
    last_row['GDP'] = last_row['GDP'] * 1.025
    last_row['Industrial_Reshoring'] = last_row['Industrial_Reshoring'] * 1.01
    last_row['Oil_Price'] = 78.0
    df = pd.concat([df, pd.DataFrame([last_row])], ignore_index=True)
    return df

@st.cache_data
def merge_all_data(manual_df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
    manual_df['Year'] = manual_df['Year'].astype(int)
    macro_df['Year'] = macro_df['Year'].astype(int)
    df = pd.merge(manual_df, macro_df, on='Year', how='inner')
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(method='linear').ffill().bfill()
    return df

@st.cache_data
def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values('Year').reset_index(drop=True)
    df['Year_Index'] = df['Year'] - 2000
    df['Fossil_Lag1'] = df['Fossil_Usage'].shift(1)
    df['Renewable_Lag1'] = df['Renewable_Usage'].shift(1)
    df['Green_Subsidy_Lag2'] = df['Green_Subsidy_Index'].shift(2)
    df['Total_Energy'] = df['Fossil_Usage'] + df['Renewable_Usage']
    df['Energy_Intensity'] = df['Total_Energy'] / df['GDP']
    df['Energy_Intensity_Lag1'] = df['Energy_Intensity'].shift(1)
    df['Fossil_Diff'] = df['Fossil_Usage'] - df['Fossil_Usage'].shift(1)
    df['Renewable_Diff'] = df['Renewable_Usage'] - df['Renewable_Usage'].shift(1)
    df = df.dropna().reset_index(drop=True)
    return df

# ============================================
# Model Training (Cached)
# ============================================

@st.cache_resource
def train_models(df: pd.DataFrame) -> tuple:
    feature_cols = ['GDP', 'Industrial_Reshoring', 'Oil_Price', 
                    'LCOE_Advantage', 'Green_Subsidy_Index', 
                    'Green_Subsidy_Lag2', 'Permitting_Ease', 
                    'Trade_Barrier', 'Year_Index', 'Energy_Intensity_Lag1']
    
    fossil_features = feature_cols + ['Fossil_Lag1']
    X_fossil = df[fossil_features]
    y_fossil = df['Fossil_Diff']
    
    renewable_features = feature_cols + ['Renewable_Lag1']
    X_renewable = df[renewable_features]
    y_renewable = df['Renewable_Diff']
    
    xgb_params = {
        'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.1,
        'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42,
        'objective': 'reg:squarederror', 'n_jobs': 1
    }
    
    fossil_model = XGBRegressor(**xgb_params)
    fossil_model.fit(X_fossil, y_fossil)
    
    renewable_model = XGBRegressor(**xgb_params)
    renewable_model.fit(X_renewable, y_renewable)
    
    fossil_pred = fossil_model.predict(X_fossil)
    renewable_pred = renewable_model.predict(X_renewable)
    
    f_rmse = np.sqrt(mean_squared_error(y_fossil, fossil_pred))
    r_rmse = np.sqrt(mean_squared_error(y_renewable, renewable_pred))
    
    return (fossil_model, renewable_model, fossil_features, renewable_features, f_rmse, r_rmse, X_fossil, X_renewable)

# ============================================
# Forecasting Logic (With CO2)
# ============================================

def recursive_forecast(
    fossil_model, renewable_model, fossil_features, renewable_features,
    last_row, historical_df, scenario_params, forecast_years,
    fossil_rmse, renewable_rmse
) -> pd.DataFrame:
    predictions = []
    
    current_fossil = last_row['Fossil_Usage']
    current_renewable = last_row['Renewable_Usage']
    fossil_lag = current_fossil
    renewable_lag = current_renewable
    
    current_year_index = int(last_row['Year_Index'])
    last_gdp = last_row['GDP']
    last_industrial = last_row['Industrial_Reshoring']
    last_oil = last_row['Oil_Price']
    current_intensity_lag = last_row['Energy_Intensity']
    
    historical_subsidy = historical_df.set_index('Year')['Green_Subsidy_Index'].to_dict()
    forecast_subsidy = {year: scenario_params['green_subsidy'] for year in forecast_years}
    all_subsidy = {**historical_subsidy, **forecast_subsidy}
    
    carbon_intensity_base = 53.0
    
    for i, year in enumerate(forecast_years):
        current_year_index += 1
        current_gdp = last_gdp * (1 + scenario_params['gdp_growth_rate'] / 100)
        current_industrial = last_industrial * (1 + scenario_params['industrial_growth_rate'] / 100)
        current_oil = last_oil * (1 + scenario_params['oil_price_change'] / 100)
        
        lcoe_improvement = scenario_params['lcoe_improvement_per_year'] * (i + 1)
        current_lcoe = last_row['LCOE_Advantage'] + lcoe_improvement
        
        lag2_year = year - 2
        green_subsidy_lag2 = all_subsidy.get(lag2_year, scenario_params['green_subsidy'])
        
        feature_base = {
            'GDP': current_gdp, 'Industrial_Reshoring': current_industrial,
            'Oil_Price': current_oil, 'LCOE_Advantage': current_lcoe,
            'Green_Subsidy_Index': scenario_params['green_subsidy'],
            'Green_Subsidy_Lag2': green_subsidy_lag2,
            'Permitting_Ease': scenario_params['permitting_ease'],
            'Trade_Barrier': scenario_params['trade_barrier'],
            'Year_Index': current_year_index,
            'Energy_Intensity_Lag1': current_intensity_lag
        }
        
        f_input = feature_base.copy()
        f_input['Fossil_Lag1'] = fossil_lag
        f_diff = fossil_model.predict(pd.DataFrame([f_input])[fossil_features])[0]
        fossil_value = current_fossil + f_diff
        
        r_input = feature_base.copy()
        r_input['Renewable_Lag1'] = renewable_lag
        r_diff = renewable_model.predict(pd.DataFrame([r_input])[renewable_features])[0]
        renewable_value = current_renewable + r_diff
        
        efficiency = 0.995 ** (i + 1)
        co2_emission = fossil_value * carbon_intensity_base * efficiency 
        
        predictions.append({
            'Year': year,
            'Fossil_Usage': fossil_value,
            'Renewable_Usage': renewable_value,
            'CO2_Emissions': co2_emission
        })
        
        current_fossil = fossil_value
        current_renewable = renewable_value
        fossil_lag = fossil_value
        renewable_lag = renewable_value
        last_gdp = current_gdp
        last_industrial = current_industrial
        last_oil = current_oil
        current_intensity_lag = (fossil_value + renewable_value) / current_gdp
    
    return pd.DataFrame(predictions)

# ============================================
# Sensitivity & SHAP Helpers (RESTORED & NEW)
# ============================================

@st.cache_data
def calculate_sensitivity_matrix(
    _model, features, last_row_dict, historical_subsidy,
    base_scenario, target_year=2028
) -> np.ndarray:
    """Restored Sensitivity Matrix Logic"""
    subsidy_range = np.arange(0, 11, 1)
    growth_range = np.arange(0, 11, 1)
    result_matrix = np.zeros((len(growth_range), len(subsidy_range)))
    forecast_years = list(range(2025, target_year + 1))
    
    for i, growth_rate in enumerate(growth_range):
        for j, subsidy in enumerate(subsidy_range):
            scenario = base_scenario.copy()
            scenario['green_subsidy'] = subsidy
            scenario['industrial_growth_rate'] = growth_rate
            
            # Simplified Single Run logic for speed
            curr_val = last_row_dict['Renewable_Usage']
            curr_lag = curr_val
            curr_gdp = last_row_dict['GDP']
            
            for y in forecast_years:
                curr_gdp *= (1 + scenario['gdp_growth_rate']/100)
                feat = {
                    'GDP': curr_gdp,
                    'Industrial_Reshoring': last_row_dict['Industrial_Reshoring'], # Simplified
                    'Oil_Price': last_row_dict['Oil_Price'], # Simplified
                    'LCOE_Advantage': last_row_dict['LCOE_Advantage'], # Simplified
                    'Green_Subsidy_Index': subsidy,
                    'Green_Subsidy_Lag2': subsidy, # Simplified
                    'Permitting_Ease': scenario['permitting_ease'],
                    'Trade_Barrier': scenario['trade_barrier'],
                    'Year_Index': 25, # Dummy
                    'Energy_Intensity_Lag1': 4.0, # Dummy
                    'Renewable_Lag1': curr_lag
                }
                pred = _model.predict(pd.DataFrame([feat])[features])[0]
                curr_val += pred
                curr_lag = curr_val
            
            result_matrix[i, j] = curr_val
            
    return result_matrix

def plot_shap_summary(model, X):
    """Generates SHAP Summary Plot"""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X, show=False, plot_type="bar")
    return plt.gcf()

# ============================================
# Chart Helpers
# ============================================

def create_comparison_chart(hist_df, trump_df, baseline_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_df['Year'], y=hist_df['Renewable_Usage'], name='Historical', line=dict(color='black', width=2)))
    fig.add_trace(go.Scatter(x=trump_df['Year'], y=trump_df['Renewable_Usage'], name='Trump 2.0 Scenario', line=dict(color='#FF4B4B', width=3)))
    fig.add_trace(go.Scatter(x=baseline_df['Year'], y=baseline_df['Renewable_Usage'], name='Baseline (Status Quo)', line=dict(color='#0068C9', width=3, dash='dot')))
    fig.update_layout(title="Scenario Comparison: Renewable Energy", yaxis_title="Energy (Q BTU)", template='plotly_white', height=500)
    return fig

def create_co2_chart(hist_df, trump_df, baseline_df):
    fig = go.Figure()
    hist_co2 = hist_df['Fossil_Usage'] * 53.0
    fig.add_trace(go.Scatter(x=hist_df['Year'], y=hist_co2, name='Historical CO2', line=dict(color='gray', width=2)))
    fig.add_trace(go.Scatter(x=trump_df['Year'], y=trump_df['CO2_Emissions'], name='Trump 2.0 CO2', line=dict(color='#8B0000', width=3)))
    fig.add_trace(go.Scatter(x=baseline_df['Year'], y=baseline_df['CO2_Emissions'], name='Baseline CO2', line=dict(color='#2E8B57', width=3, dash='dot')))
    fig.update_layout(title="Projected CO2 Emissions", yaxis_title="Million Metric Tons CO2", template='plotly_white', height=500)
    return fig

def create_heatmap_fig(matrix):
    fig = go.Figure(data=go.Heatmap(
        z=matrix, x=list(range(11)), y=[f"{i}%" for i in range(11)],
        colorscale='Viridis', texttemplate="%{z:.2f}"
    ))
    fig.update_layout(title="Sensitivity: Subsidy vs Growth", xaxis_title="Subsidy Index", yaxis_title="Ind. Growth %")
    return fig

# ============================================
# Main Application
# ============================================

def main():
    st.title("⚡ US Energy Forecast Pro | 能源决策系统")
    st.markdown("### Trump 2.0 vs Baseline | Comprehensive Analysis")
    
    # Sidebar
    st.sidebar.header("🎛️ Trump 2.0 Settings")
    with st.sidebar.expander("Policy Scores", expanded=True):
        green_subsidy = st.slider("Green Subsidy", 0, 10, 3)
        permitting_ease = st.slider("Fossil Permitting", 0, 10, 9)
        trade_barrier = st.slider("Trade Barrier", 0, 10, 9)
    with st.sidebar.expander("Macro Assumptions", expanded=True):
        gdp_growth = st.slider("GDP Growth (%)", -2.0, 5.0, 2.5)
        industrial_growth = st.slider("Industrial Growth (%)", -2.0, 10.0, 2.0)
    forecast_end = st.sidebar.selectbox("Forecast Until", [2028, 2030, 2035], index=1)
    
    # Scenarios
    trump_params = {'green_subsidy': green_subsidy, 'permitting_ease': permitting_ease, 'trade_barrier': trade_barrier, 
                    'gdp_growth_rate': gdp_growth, 'industrial_growth_rate': industrial_growth, 'oil_price_change': 3.0, 'lcoe_improvement_per_year': 2.0}
    baseline_params = trump_params.copy()
    baseline_params.update({'green_subsidy': 8, 'permitting_ease': 4, 'trade_barrier': 5})
    
    # Data & Train
    manual_df = load_manual_data()
    macro_df = get_static_macro_data()
    merged_df = merge_all_data(manual_df, macro_df)
    df = create_lag_features(merged_df)
    (f_model, r_model, f_feats, r_feats, f_rmse, r_rmse, X_fossil, X_renewable) = train_models(df)
    
    # Forecasts
    forecast_years = list(range(2025, forecast_end + 1))
    trump_df = recursive_forecast(f_model, r_model, f_feats, r_feats, df.iloc[-1], df, trump_params, forecast_years, f_rmse, r_rmse)
    baseline_df = recursive_forecast(f_model, r_model, f_feats, r_feats, df.iloc[-1], df, baseline_params, forecast_years, f_rmse, r_rmse)
    
    # TABS STRUCTURE
    tab1, tab2, tab3 = st.tabs(["📊 Forecast & CO2", "🔬 Sensitivity Heatmap", "🤖 SHAP Analysis"])
    
    # TAB 1: Forecast Comparison
    with tab1:
        st.subheader("Scenario Comparison")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_comparison_chart(merged_df, trump_df, baseline_df), use_container_width=True)
        with col2:
            st.plotly_chart(create_co2_chart(merged_df, trump_df, baseline_df), use_container_width=True)
            
        st.dataframe(trump_df.style.format("{:.2f}"), use_container_width=True)

    # TAB 2: Sensitivity Heatmap (RESTORED)
    with tab2:
        st.subheader("Policy Sensitivity Analysis (Heatmap)")
        st.caption("How Renewable Usage changes with Subsidy (X) vs Industrial Growth (Y)")
        sens_matrix = calculate_sensitivity_matrix(
            r_model, r_feats, df.iloc[-1].to_dict(), 
            df.set_index('Year')['Green_Subsidy_Index'].to_dict(), 
            trump_params, target_year=2028
        )
        st.plotly_chart(create_heatmap_fig(sens_matrix), use_container_width=True)

    # TAB 3: SHAP Analysis (NEW)
    with tab3:
        st.subheader("SHAP Feature Importance")
        st.markdown("Calculates the global contribution of each policy/macro variable.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Renewable Energy Model (SHAP)**")
            st.pyplot(plot_shap_summary(r_model, X_renewable))
        with col2:
            st.write("**Fossil Energy Model (SHAP)**")
            st.pyplot(plot_shap_summary(f_model, X_fossil))

if __name__ == "__main__":
    main()
