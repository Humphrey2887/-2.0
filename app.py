"""
US Energy Consumption Forecast | 美国能源消费预测 - app.py (Auto-Load Sensitivity Version)
XGBoost Forecasting for Trump 2.0 Scenario | XGBoost预测 Trump 2.0情景分析

Updates:
- Sensitivity Analysis now runs automatically on page load (No button required).
- 敏感性分析现在会自动运行，无需点击按钮。
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import warnings

# 忽略警告信息
warnings.filterwarnings('ignore')

# Page Configuration | 页面配置
st.set_page_config(
    page_title="Energy Forecast | 能源预测",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# Data Loading & Processing | 数据加载与处理
# ============================================

@st.cache_data
def load_manual_data(filepath: str = "manual_data.csv") -> pd.DataFrame:
    """Load processed manual data | 加载处理后的手动数据"""
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        st.error(f"File not found | 找不到文件: {filepath}. Please run process_data.py first | 请先运行 process_data.py")
        st.stop()


@st.cache_data
def get_static_macro_data() -> pd.DataFrame:
    """
    Returns embedded REAL historical macro data (2000-2023).
    Bypasses FRED API blocking issues completely.
    """
    data = {
        'Year': list(range(2000, 2024)),
        # 真实美国GDP数据 (Billions USD)
        'GDP': [
            10252, 10581, 10936, 11458, 12213, 13036, 13814, 14451, 14712, 14448, 
            14992, 15542, 16197, 16784, 17521, 18219, 18707, 19485, 20527, 21372, 
            20893, 22996, 25462, 27360
        ],
        # 真实工业产出指数 (INDPRO, 2017=100)
        'Industrial_Reshoring': [
            92.8, 89.4, 89.7, 90.9, 93.3, 96.5, 98.6, 100.0, 96.3, 85.0, 
            90.6, 93.6, 96.6, 98.4, 101.3, 100.6, 99.4, 100.0, 103.1, 102.4, 
            95.4, 100.4, 103.7, 103.0
        ],
        # 真实WTI原油价格 (DCOILWTICO)
        'Oil_Price': [
            30.3, 25.9, 26.1, 31.1, 41.4, 56.5, 66.1, 72.3, 99.6, 61.7, 
            79.4, 94.8, 94.1, 97.9, 93.1, 48.7, 43.2, 50.8, 65.2, 56.9, 
            39.2, 68.1, 94.4, 77.6
        ]
    }
    
    df = pd.DataFrame(data)
    # 简单插值到2024
    last_row = df.iloc[-1].copy()
    last_row['Year'] = 2024
    last_row['GDP'] = last_row['GDP'] * 1.025
    last_row['Industrial_Reshoring'] = last_row['Industrial_Reshoring'] * 1.01
    last_row['Oil_Price'] = 78.0
    
    df = pd.concat([df, pd.DataFrame([last_row])], ignore_index=True)
    return df


@st.cache_data
def merge_all_data(manual_df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
    """Merge manual and macro data | 合并手动数据与宏观数据"""
    manual_df['Year'] = manual_df['Year'].astype(int)
    macro_df['Year'] = macro_df['Year'].astype(int)
    
    df = pd.merge(manual_df, macro_df, on='Year', how='inner')
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(method='linear')
    df[numeric_cols] = df[numeric_cols].ffill().bfill()
    
    return df


@st.cache_data
def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features | 创建特征"""
    df = df.copy()
    df = df.sort_values('Year').reset_index(drop=True)
    
    df['Year_Index'] = df['Year'] - 2000
    df['Year_Index'] = df['Year_Index'].astype(int)
    
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
# Model Training (Cached for Speed) | 模型训练(缓存加速)
# ============================================

@st.cache_resource
def train_models(df: pd.DataFrame) -> tuple:
    """Train XGBoost models | 训练模型"""
    feature_cols = ['GDP', 'Industrial_Reshoring', 'Oil_Price', 
                    'LCOE_Advantage', 'Green_Subsidy_Index', 
                    'Green_Subsidy_Lag2',
                    'Permitting_Ease', 'Trade_Barrier', 'Year_Index',
                    'Energy_Intensity_Lag1']
    
    fossil_features = feature_cols + ['Fossil_Lag1']
    X_fossil = df[fossil_features]
    y_fossil = df['Fossil_Diff']
    
    renewable_features = feature_cols + ['Renewable_Lag1']
    X_renewable = df[renewable_features]
    y_renewable = df['Renewable_Diff']
    
    xgb_params = {
        'n_estimators': 100,
        'max_depth': 4,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'objective': 'reg:squarederror',
        'n_jobs': 1
    }
    
    fossil_model = XGBRegressor(**xgb_params)
    fossil_model.fit(X_fossil, y_fossil)
    
    renewable_model = XGBRegressor(**xgb_params)
    renewable_model.fit(X_renewable, y_renewable)
    
    fossil_pred_train = fossil_model.predict(X_fossil)
    fossil_rmse = np.sqrt(mean_squared_error(y_fossil, fossil_pred_train))
    
    renewable_pred_train = renewable_model.predict(X_renewable)
    renewable_rmse = np.sqrt(mean_squared_error(y_renewable, renewable_pred_train))
    
    return (fossil_model, renewable_model, fossil_features, renewable_features,
            fossil_rmse, renewable_rmse)


def recursive_forecast(
    fossil_model, renewable_model,
    fossil_features, renewable_features,
    last_row, historical_df,
    scenario_params, forecast_years,
    fossil_rmse, renewable_rmse
) -> pd.DataFrame:
    """Recursive forecasting | 递归预测"""
    predictions = []
    
    current_fossil = last_row['Fossil_Usage']
    current_renewable = last_row['Renewable_Usage']
    
    fossil_lag = last_row['Fossil_Usage']
    renewable_lag = last_row['Renewable_Usage']
    
    current_year_index = int(last_row['Year_Index'])
    
    last_gdp = last_row['GDP']
    last_industrial = last_row['Industrial_Reshoring']
    last_oil = last_row['Oil_Price']
    
    current_intensity_lag = last_row['Energy_Intensity']
    
    historical_subsidy = historical_df.set_index('Year')['Green_Subsidy_Index'].to_dict()
    forecast_subsidy = {year: scenario_params['green_subsidy'] for year in forecast_years}
    all_subsidy = {**historical_subsidy, **forecast_subsidy}
    
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
            'GDP': current_gdp,
            'Industrial_Reshoring': current_industrial,
            'Oil_Price': current_oil,
            'LCOE_Advantage': current_lcoe,
            'Green_Subsidy_Index': scenario_params['green_subsidy'],
            'Green_Subsidy_Lag2': green_subsidy_lag2,
            'Permitting_Ease': scenario_params['permitting_ease'],
            'Trade_Barrier': scenario_params['trade_barrier'],
            'Year_Index': current_year_index,
            'Energy_Intensity_Lag1': current_intensity_lag
        }
        
        fossil_input = feature_base.copy()
        fossil_input['Fossil_Lag1'] = fossil_lag
        X_fossil = pd.DataFrame([fossil_input])[fossil_features]
        fossil_diff_pred = fossil_model.predict(X_fossil)[0]
        fossil_value = current_fossil + fossil_diff_pred
        
        renewable_input = feature_base.copy()
        renewable_input['Renewable_Lag1'] = renewable_lag
        X_renewable = pd.DataFrame([renewable_input])[renewable_features]
        renewable_diff_pred = renewable_model.predict(X_renewable)[0]
        renewable_value = current_renewable + renewable_diff_pred
        
        fossil_cumulative_std = fossil_rmse * np.sqrt(i + 1)
        renewable_cumulative_std = renewable_rmse * np.sqrt(i + 1)
        
        fossil_upper = fossil_value + 1.96 * fossil_cumulative_std
        fossil_lower = fossil_value - 1.96 * fossil_cumulative_std
        renewable_upper = renewable_value + 1.96 * renewable_cumulative_std
        renewable_lower = renewable_value - 1.96 * renewable_cumulative_std
        
        predictions.append({
            'Year': year,
            'Fossil_Usage': fossil_value,
            'Fossil_Upper': fossil_upper,
            'Fossil_Lower': fossil_lower,
            'Renewable_Usage': renewable_value,
            'Renewable_Upper': renewable_upper,
            'Renewable_Lower': renewable_lower,
            'GDP': current_gdp,
            'Year_Index': current_year_index
        })
        
        current_fossil = fossil_value
        current_renewable = renewable_value
        fossil_lag = fossil_value
        renewable_lag = renewable_value
        last_gdp = current_gdp
        last_industrial = current_industrial
        last_oil = current_oil
        
        total_energy_pred = fossil_value + renewable_value
        current_intensity_lag = total_energy_pred / current_gdp
    
    return pd.DataFrame(predictions)


# ============================================
# Visualization | 可视化
# ============================================

def create_forecast_chart_with_ci(historical_df, forecast_df, energy_type='both'):
    """Create chart | 创建图表"""
    fig = go.Figure()
    
    colors = {
        'fossil_hist': '#8B4513', 'fossil_pred': '#D2691E', 'fossil_ci': 'rgba(210, 105, 30, 0.2)',
        'renewable_hist': '#228B22', 'renewable_pred': '#32CD32', 'renewable_ci': 'rgba(50, 205, 50, 0.2)'
    }
    
    last_hist_year = historical_df['Year'].max()
    forecast_x = [last_hist_year] + forecast_df['Year'].tolist()
    
    if energy_type in ['fossil', 'both']:
        # Fossil Logic
        fig.add_trace(go.Scatter(
            x=historical_df['Year'], y=historical_df['Fossil_Usage'],
            mode='lines+markers', name='Fossil (Historical)',
            line=dict(color=colors['fossil_hist'], width=2)
        ))
        
        last_val = historical_df[historical_df['Year'] == last_hist_year]['Fossil_Usage'].values[0]
        upper_y = [last_val] + forecast_df['Fossil_Upper'].tolist()
        lower_y = [last_val] + forecast_df['Fossil_Lower'].tolist()
        pred_y = [last_val] + forecast_df['Fossil_Usage'].tolist()
        
        fig.add_trace(go.Scatter(
            x=forecast_x, y=upper_y, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=forecast_x, y=lower_y, mode='lines', line=dict(width=0), fill='tonexty',
            fillcolor=colors['fossil_ci'], name='Fossil 95% CI', hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=forecast_x, y=pred_y, mode='lines+markers', name='Fossil (Forecast)',
            line=dict(color=colors['fossil_pred'], width=2, dash='dash')
        ))
    
    if energy_type in ['renewable', 'both']:
        # Renewable Logic
        fig.add_trace(go.Scatter(
            x=historical_df['Year'], y=historical_df['Renewable_Usage'],
            mode='lines+markers', name='Renewable (Historical)',
            line=dict(color=colors['renewable_hist'], width=2)
        ))
        
        last_val = historical_df[historical_df['Year'] == last_hist_year]['Renewable_Usage'].values[0]
        upper_y = [last_val] + forecast_df['Renewable_Upper'].tolist()
        lower_y = [last_val] + forecast_df['Renewable_Lower'].tolist()
        pred_y = [last_val] + forecast_df['Renewable_Usage'].tolist()
        
        fig.add_trace(go.Scatter(
            x=forecast_x, y=upper_y, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=forecast_x, y=lower_y, mode='lines', line=dict(width=0), fill='tonexty',
            fillcolor=colors['renewable_ci'], name='Renewable 95% CI', hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=forecast_x, y=pred_y, mode='lines+markers', name='Renewable (Forecast)',
            line=dict(color=colors['renewable_pred'], width=2, dash='dash')
        ))
    
    fig.add_vline(x=last_hist_year, line_dash="dot", line_color="gray")
    fig.update_layout(
        title='<b>US Energy Forecast | 美国能源预测</b>',
        xaxis_title='Year', yaxis_title='Energy (Q BTU)',
        template='plotly_white', height=550
    )
    return fig


def create_sensitivity_heatmap(sensitivity_matrix, target_year):
    """Create heatmap | 创建热力图"""
    z_min = float(np.min(sensitivity_matrix))
    z_max = float(np.max(sensitivity_matrix))
    z_delta = z_max - z_min
    
    x_values = list(range(11))
    y_values = [f"{i}%" for i in range(11)]
    
    fig = go.Figure(data=go.Heatmap(
        z=sensitivity_matrix, x=x_values, y=y_values,
        zmin=z_min, zmax=z_max, colorscale='Viridis',
        texttemplate="%{z:.3f}", hovertemplate="Subsidy: %{x}<br>Growth: %{y}<br>Usage: %{z:.4f}<extra></extra>"
    ))
    
    fig.update_layout(
        title=f"Sensitivity Analysis {target_year} | 敏感性分析",
        xaxis_title="Green Subsidy Index | 补贴指数",
        yaxis_title="Industrial Growth (%) | 工业增长率",
        height=600
    )
    return fig, z_min, z_max, z_delta


def create_feature_importance_chart(fossil_model, renewable_model, feature_names):
    """Create feature importance chart | 创建特征重要性图表"""
    base_features = [f for f in feature_names if 'Lag' not in f]
    n_base = len(base_features)
    
    fossil_imp = fossil_model.feature_importances_[:n_base]
    renewable_imp = renewable_model.feature_importances_[:n_base]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Fossil Model', x=base_features, y=fossil_imp, marker_color='#8B4513'))
    fig.add_trace(go.Bar(name='Renewable Model', x=base_features, y=renewable_imp, marker_color='#228B22'))
    
    fig.update_layout(title='Feature Importance | 特征重要性', barmode='group', template='plotly_white', height=400)
    return fig


# ============================================
# Sensitivity Analysis Logic (Cached) | 敏感性分析逻辑
# ============================================

@st.cache_data
def calculate_sensitivity(
    _fossil_model, _renewable_model,
    _fossil_features, _renewable_features,
    _last_row_tuple, _historical_subsidy_tuple,
    base_scenario, target_year=2028
) -> np.ndarray:
    """Calculate sensitivity matrix | 计算敏感性分析矩阵"""
    last_row_dict = dict(_last_row_tuple)
    historical_subsidy = dict(_historical_subsidy_tuple)
    renewable_features = list(_renewable_features)
    
    subsidy_range = np.arange(0, 11, 1)
    growth_range = np.arange(0, 11, 1)
    
    result_matrix = np.zeros((len(growth_range), len(subsidy_range)))
    
    forecast_years = list(range(2025, target_year + 1))
    
    for i, growth_rate in enumerate(growth_range):
        for j, subsidy in enumerate(subsidy_range):
            current_scenario = base_scenario.copy()
            current_scenario['green_subsidy'] = subsidy
            current_scenario['industrial_growth_rate'] = growth_rate
            
            renewable_value = _run_single_forecast(
                _renewable_model, renewable_features,
                last_row_dict, historical_subsidy,
                current_scenario, forecast_years
            )
            
            result_matrix[i, j] = renewable_value
    
    return result_matrix


def _run_single_forecast(
    model, features, last_row_dict, historical_subsidy,
    scenario, forecast_years
) -> float:
    """Run single forecast (Helper) | 运行单次预测（辅助函数）"""
    current_value = last_row_dict['Renewable_Usage']
    renewable_lag = last_row_dict['Renewable_Usage']
    current_year_index = int(last_row_dict['Year_Index'])
    
    last_gdp = last_row_dict['GDP']
    last_industrial = last_row_dict['Industrial_Reshoring']
    last_oil = last_row_dict['Oil_Price']
    
    current_intensity_lag = last_row_dict.get('Energy_Intensity', 
        (last_row_dict['Fossil_Usage'] + last_row_dict['Renewable_Usage']) / last_gdp)
    fossil_estimate = last_row_dict['Fossil_Usage']
    
    forecast_subsidy = {year: scenario['green_subsidy'] for year in forecast_years}
    all_subsidy = {**historical_subsidy, **forecast_subsidy}
    
    for i, year in enumerate(forecast_years):
        current_year_index += 1
        
        current_gdp = last_gdp * (1 + scenario['gdp_growth_rate'] / 100)
        current_industrial = last_industrial * (1 + scenario['industrial_growth_rate'] / 100)
        current_oil = last_oil * (1 + scenario['oil_price_change'] / 100)
        
        lcoe_improvement = scenario['lcoe_improvement_per_year'] * (i + 1)
        current_lcoe = last_row_dict['LCOE_Advantage'] + lcoe_improvement
        
        lag2_year = year - 2
        green_subsidy_lag2 = all_subsidy.get(lag2_year, scenario['green_subsidy'])
        
        feature_input = {
            'GDP': current_gdp,
            'Industrial_Reshoring': current_industrial,
            'Oil_Price': current_oil,
            'LCOE_Advantage': current_lcoe,
            'Green_Subsidy_Index': scenario['green_subsidy'],
            'Green_Subsidy_Lag2': green_subsidy_lag2,
            'Permitting_Ease': scenario['permitting_ease'],
            'Trade_Barrier': scenario['trade_barrier'],
            'Year_Index': current_year_index,
            'Energy_Intensity_Lag1': current_intensity_lag,
            'Renewable_Lag1': renewable_lag
        }
        
        X = pd.DataFrame([feature_input])[features]
        diff_pred = model.predict(X)[0]
        current_value = current_value + diff_pred
        
        renewable_lag = current_value
        last_gdp = current_gdp
        last_industrial = current_industrial
        last_oil = current_oil
        
        total_energy_estimate = fossil_estimate + current_value
        current_intensity_lag = total_energy_estimate / current_gdp
    
    return current_value


# ============================================
# Main Application | 主应用
# ============================================

def main():
    st.title("⚡ US Energy Forecast Pro | 美国能源消费预测")
    st.markdown("### Trump 2.0 Scenario Analysis")
    
    # Sidebar
    st.sidebar.header("🎛️ Settings")
    
    with st.sidebar.expander("Policy Scores | 政策评分", expanded=True):
        green_subsidy = st.slider("Green Subsidy | 补贴", 0, 10, 3)
        permitting_ease = st.slider("Permitting Ease | 审批", 0, 10, 9)
        trade_barrier = st.slider("Trade Barrier | 关税", 0, 10, 9)
    
    with st.sidebar.expander("Macro Assumptions | 宏观假设", expanded=True):
        gdp_growth = st.slider("GDP Growth (%)", -2.0, 5.0, 2.5)
        industrial_growth = st.slider("Industrial Growth (%)", -2.0, 10.0, 2.0)
        oil_price_change = st.slider("Oil Price Change (%)", -20.0, 20.0, 3.0)
    
    lcoe_improvement = st.sidebar.slider("LCOE Improvement", 0.0, 10.0, 2.0)
    forecast_end = st.sidebar.selectbox("Forecast Until", [2026, 2028, 2030], index=1)
    
    scenario_params = {
        'green_subsidy': green_subsidy, 'permitting_ease': permitting_ease,
        'trade_barrier': trade_barrier, 'gdp_growth_rate': gdp_growth,
        'industrial_growth_rate': industrial_growth, 'oil_price_change': oil_price_change,
        'lcoe_improvement_per_year': lcoe_improvement
    }
    
    # Execution
    with st.spinner("Loading Data..."):
        manual_df = load_manual_data()
        macro_df = get_static_macro_data() 
        merged_df = merge_all_data(manual_df, macro_df)
        df = create_lag_features(merged_df)
    
    # Metrics
    col1, col2 = st.columns(2)
    col1.metric("2024 Fossil", f"{df['Fossil_Usage'].iloc[-1]:.1f} Q BTU")
    col2.metric("2024 Renewable", f"{df['Renewable_Usage'].iloc[-1]:.1f} Q BTU")
    
    # Model & Forecast
    (f_model, r_model, f_feats, r_feats, f_rmse, r_rmse) = train_models(df)
        
    forecast_years = list(range(2025, forecast_end + 1))
    forecast_df = recursive_forecast(
        f_model, r_model, f_feats, r_feats,
        df.iloc[-1], df, scenario_params, forecast_years, f_rmse, r_rmse
    )
    
    # Visualization
    st.subheader("Forecast Results")
    energy_type = st.radio("View", ['both', 'fossil', 'renewable'], horizontal=True)
    st.plotly_chart(create_forecast_chart_with_ci(merged_df, forecast_df, energy_type), use_container_width=True)
    
    st.dataframe(forecast_df[['Year', 'Fossil_Usage', 'Renewable_Usage']].style.format("{:.2f}"), use_container_width=True)
    
    # Sensitivity Analysis - NOW AUTOMATIC (No Button)
    st.markdown("---")
    st.subheader("Sensitivity Analysis (Microscope Mode)")
    
    # 🟢 自动运行敏感性分析 (已移除按钮)
    sens_matrix = calculate_sensitivity(
        f_model, r_model, tuple(f_feats), tuple(r_feats),
        tuple(df.iloc[-1].items()), 
        tuple(df.set_index('Year')['Green_Subsidy_Index'].items()),
        scenario_params
    )
    fig_hm, _, _, _ = create_sensitivity_heatmap(sens_matrix, 2028)
    st.plotly_chart(fig_hm, use_container_width=True)

    # Feature Importance
    st.markdown("---")
    st.plotly_chart(create_feature_importance_chart(f_model, r_model, f_feats), use_container_width=True)

if __name__ == "__main__":
    main()
