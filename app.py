"""
US Energy Consumption Forecast | 美国能源消费预测 - app.py (Pro Bilingual Version)
XGBoost Forecasting for Trump 2.0 Scenario | XGBoost预测 Trump 2.0情景分析

Pro Features | 专业版功能:
1. Uncertainty Quantification | 不确定性量化 (Confidence Intervals | 置信区间)
2. Policy Lag Effects | 政策滞后效应 (2-Year Transmission | 2年传导)
3. Sensitivity Analysis Heatmap | 敏感性分析热力图
4. Energy Intensity Feature | 能源强度特征
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import warnings
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


@st.cache_data(ttl=3600)
def fetch_fred_data(start_year: int = 2000, end_year: int = 2024) -> pd.DataFrame:
    """
    Fetch macro data from FRED | 从FRED获取宏观经济数据
    - GDP: Gross Domestic Product | 国内生产总值
    - Industrial_Reshoring: Industrial Production Index | 工业生产指数
    - Oil_Price: Crude Oil Price | 原油价格
    """
    try:
        import pandas_datareader.data as web
        from datetime import datetime
        
        start = datetime(start_year, 1, 1)
        end = datetime(end_year, 12, 31)
        
        series_map = {
            'GDP': 'GDP',
            'Industrial_Reshoring': 'INDPRO',
            'Oil_Price': 'DCOILWTICO'
        }
        
        macro_data = {}
        
        for name, series_id in series_map.items():
            try:
                data = web.DataReader(series_id, 'fred', start, end)
                annual_data = data.resample('YE').mean()
                annual_data.index = annual_data.index.year
                macro_data[name] = annual_data[series_id]
            except Exception as e:
                st.warning(f"Cannot fetch | 无法获取 {name} ({series_id}): {e}")
                macro_data[name] = None
        
        df = pd.DataFrame(macro_data)
        df.index.name = 'Year'
        df = df.reset_index()
        
        return df
        
    except ImportError:
        st.warning("pandas_datareader not installed, using mock data | pandas_datareader未安装，使用模拟数据")
        return generate_mock_macro_data(start_year, end_year)
    except Exception as e:
        st.warning(f"FRED data fetch failed, using mock data | FRED数据获取失败，使用模拟数据: {e}")
        return generate_mock_macro_data(start_year, end_year)


def generate_mock_macro_data(start_year: int, end_year: int) -> pd.DataFrame:
    """Generate mock macro data | 生成模拟宏观数据"""
    np.random.seed(42)
    years = list(range(start_year, end_year + 1))
    n = len(years)
    
    base_gdp = 10000
    gdp_growth = np.cumsum(np.random.normal(500, 200, n))
    gdp = base_gdp + gdp_growth
    
    industrial = 90 + np.cumsum(np.random.normal(1.5, 1, n))
    
    oil_base = 30
    oil_prices = oil_base + 40 * np.sin(np.linspace(0, 4*np.pi, n)) + np.random.normal(0, 10, n)
    oil_prices = np.clip(oil_prices, 20, 150)
    
    return pd.DataFrame({
        'Year': years,
        'GDP': gdp,
        'Industrial_Reshoring': industrial,
        'Oil_Price': oil_prices
    })


def merge_all_data(manual_df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
    """Merge manual and macro data | 合并手动数据与宏观数据"""
    df = pd.merge(manual_df, macro_df, on='Year', how='inner')
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(method='linear')
    df[numeric_cols] = df[numeric_cols].ffill().bfill()
    
    return df


def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create lag features, time trend, policy lag, energy intensity and diff targets |
    创建滞后特征、时间趋势、政策滞后、能源强度和差分目标"""
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
# Model Training & Forecasting | 模型训练与预测
# ============================================

def train_models(df: pd.DataFrame) -> tuple:
    """
    Train XGBoost models - Predict YoY Change (Diff) | 训练XGBoost模型 - 预测年度变化量
    
    Returns:
        (fossil_model, renewable_model, fossil_features, renewable_features, 
         fossil_rmse, renewable_rmse)
    """
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
        'objective': 'reg:squarederror'
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
    fossil_model: XGBRegressor,
    renewable_model: XGBRegressor,
    fossil_features: list,
    renewable_features: list,
    last_row: pd.Series,
    historical_df: pd.DataFrame,
    scenario_params: dict,
    forecast_years: list,
    fossil_rmse: float,
    renewable_rmse: float
) -> pd.DataFrame:
    """
    Recursive forecasting with diff modeling + confidence intervals + policy lag |
    递归预测：差分建模 + 置信区间 + 政策滞后
    """
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
    
    fossil_cumulative_std = 0
    renewable_cumulative_std = 0
    
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
            'Fossil_Diff': fossil_diff_pred,
            'Renewable_Diff': renewable_diff_pred,
            'Green_Subsidy_Lag2': green_subsidy_lag2,
            'GDP': current_gdp,
            'Industrial_Reshoring': current_industrial,
            'Oil_Price': current_oil,
            'LCOE_Advantage': current_lcoe,
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
# Sensitivity Analysis | 敏感性分析
# ============================================

@st.cache_data
def calculate_sensitivity(
    _fossil_model,
    _renewable_model,
    _fossil_features: tuple,
    _renewable_features: tuple,
    _last_row_tuple: tuple,
    _historical_subsidy_tuple: tuple,
    base_scenario: dict,
    target_year: int = 2028
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
    model,
    features: list,
    last_row_dict: dict,
    historical_subsidy: dict,
    scenario: dict,
    forecast_years: list
) -> float:
    """Run single renewable forecast | 运行单次可再生能源预测"""
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
# Visualization | 可视化
# ============================================

def create_forecast_chart_with_ci(
    historical_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    energy_type: str = 'both'
) -> go.Figure:
    """Create forecast chart with confidence intervals | 创建带置信区间的预测图表"""
    fig = go.Figure()
    
    colors = {
        'fossil_hist': '#8B4513',
        'fossil_pred': '#D2691E',
        'fossil_ci': 'rgba(210, 105, 30, 0.2)',
        'renewable_hist': '#228B22',
        'renewable_pred': '#32CD32',
        'renewable_ci': 'rgba(50, 205, 50, 0.2)'
    }
    
    last_hist_year = historical_df['Year'].max()
    
    if energy_type in ['fossil', 'both']:
        fig.add_trace(go.Scatter(
            x=historical_df['Year'],
            y=historical_df['Fossil_Usage'],
            mode='lines+markers',
            name='Fossil (Historical) | 化石能源 (历史)',
            line=dict(color=colors['fossil_hist'], width=2),
            marker=dict(size=6)
        ))
        
        forecast_x = [last_hist_year] + forecast_df['Year'].tolist()
        last_fossil = historical_df[historical_df['Year'] == last_hist_year]['Fossil_Usage'].values[0]
        
        upper_y = [last_fossil] + forecast_df['Fossil_Upper'].tolist()
        lower_y = [last_fossil] + forecast_df['Fossil_Lower'].tolist()
        
        fig.add_trace(go.Scatter(
            x=forecast_x, y=upper_y, mode='lines', line=dict(width=0),
            showlegend=False, hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_x, y=lower_y, mode='lines', line=dict(width=0),
            fill='tonexty', fillcolor=colors['fossil_ci'],
            name='Fossil 95%CI | 化石能源置信区间', hoverinfo='skip'
        ))
        
        forecast_y = [last_fossil] + forecast_df['Fossil_Usage'].tolist()
        fig.add_trace(go.Scatter(
            x=forecast_x, y=forecast_y,
            mode='lines+markers',
            name='Fossil (Forecast) | 化石能源 (预测)',
            line=dict(color=colors['fossil_pred'], width=2, dash='dash'),
            marker=dict(size=8, symbol='diamond')
        ))
    
    if energy_type in ['renewable', 'both']:
        fig.add_trace(go.Scatter(
            x=historical_df['Year'],
            y=historical_df['Renewable_Usage'],
            mode='lines+markers',
            name='Renewable (Historical) | 可再生能源 (历史)',
            line=dict(color=colors['renewable_hist'], width=2),
            marker=dict(size=6)
        ))
        
        forecast_x = [last_hist_year] + forecast_df['Year'].tolist()
        last_renewable = historical_df[historical_df['Year'] == last_hist_year]['Renewable_Usage'].values[0]
        
        upper_y = [last_renewable] + forecast_df['Renewable_Upper'].tolist()
        lower_y = [last_renewable] + forecast_df['Renewable_Lower'].tolist()
        
        fig.add_trace(go.Scatter(
            x=forecast_x, y=upper_y, mode='lines', line=dict(width=0),
            showlegend=False, hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_x, y=lower_y, mode='lines', line=dict(width=0),
            fill='tonexty', fillcolor=colors['renewable_ci'],
            name='Renewable 95%CI | 可再生能源置信区间', hoverinfo='skip'
        ))
        
        forecast_y = [last_renewable] + forecast_df['Renewable_Usage'].tolist()
        fig.add_trace(go.Scatter(
            x=forecast_x, y=forecast_y,
            mode='lines+markers',
            name='Renewable (Forecast) | 可再生能源 (预测)',
            line=dict(color=colors['renewable_pred'], width=2, dash='dash'),
            marker=dict(size=8, symbol='diamond')
        ))
    
    fig.add_vline(
        x=last_hist_year, 
        line_dash="dot", 
        line_color="gray",
        annotation_text="Forecast Start | 预测起点",
        annotation_position="top"
    )
    
    fig.update_layout(
        title=dict(
            text='<b>US Energy Consumption Trend & Forecast | 美国能源消费趋势与预测</b><br>'
                 '<sup>Trump 2.0 Scenario | XGBoost + 95% Confidence Interval | 置信区间</sup>',
            x=0.5,
            font=dict(size=18)
        ),
        xaxis_title='Year | 年份',
        yaxis_title='Energy Consumption (Quadrillion BTU) | 能源消费',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='x unified',
        template='plotly_white',
        height=550
    )
    
    return fig


def create_sensitivity_heatmap(
    sensitivity_matrix: np.ndarray,
    target_year: int = 2028
) -> tuple:
    """Create sensitivity heatmap - Microscope Mode | 创建敏感性热力图 - 显微镜模式"""
    subsidy_labels = [str(i) for i in range(11)]
    growth_labels = [f"{i}%" for i in range(11)]
    
    z_min = np.min(sensitivity_matrix)
    z_max = np.max(sensitivity_matrix)
    z_delta = z_max - z_min
    
    if z_delta < 0.0001:
        z_min -= 0.001
        z_max += 0.001
        z_delta = z_max - z_min
    
    fig = go.Figure(data=go.Heatmap(
        z=sensitivity_matrix,
        x=subsidy_labels,
        y=growth_labels,
        colorscale='Viridis',
        zmin=z_min,
        zmax=z_max,
        zauto=False,
        colorbar=dict(
            title=f'{target_year}<br>Renewable<br>可再生能源<br>(Q BTU)',
            titleside='right',
            tickformat='.3f'
        ),
        hovertemplate=(
            '<b>Subsidy Index | 补贴指数</b>: %{x}<br>'
            '<b>Growth Rate | 增长率</b>: %{y}<br>'
            '<b>Renewable | 可再生能源</b>: %{z:.5f} Q BTU<br>'
            '<extra></extra>'
        )
    ))
    
    for i in range(sensitivity_matrix.shape[0]):
        for j in range(sensitivity_matrix.shape[1]):
            value = sensitivity_matrix[i, j]
            relative_pos = (value - z_min) / z_delta if z_delta > 0 else 0.5
            text_color = 'white' if relative_pos < 0.5 else 'black'
            
            fig.add_annotation(
                x=j, y=i,
                text=f"{value:.2f}",
                showarrow=False,
                font=dict(size=8, color=text_color)
            )
    
    fig.update_layout(
        title=dict(
            text=f'<b>🔬 Policy vs Growth Sensitivity | 政策与增长敏感性分析 (Microscope Mode | 显微镜模式)</b><br>'
                 f'<sup>{target_year} Renewable Forecast | 可再生能源预测值 | Color Range Optimized | 颜色范围已优化</sup>',
            x=0.5,
            font=dict(size=16)
        ),
        xaxis_title='Green Subsidy Index | 绿色补贴指数',
        yaxis_title='Industrial Reshoring Growth | 工业回流增长率',
        template='plotly_white',
        height=500
    )
    
    return fig, z_min, z_max, z_delta


def create_feature_importance_chart(
    fossil_model: XGBRegressor,
    renewable_model: XGBRegressor,
    feature_names: list
) -> go.Figure:
    """Create feature importance chart | 创建特征重要性图表"""
    base_features = [f for f in feature_names if 'Lag' not in f]
    n_base = len(base_features)
    
    fossil_importance = fossil_model.feature_importances_[:n_base]
    renewable_importance = renewable_model.feature_importances_[:n_base]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Fossil Model | 化石能源模型',
        x=base_features,
        y=fossil_importance,
        marker_color='#8B4513'
    ))
    
    fig.add_trace(go.Bar(
        name='Renewable Model | 可再生能源模型',
        x=base_features,
        y=renewable_importance,
        marker_color='#228B22'
    ))
    
    fig.update_layout(
        title='<b>Feature Importance Comparison | 特征重要性对比</b><br>'
              '<sup>Including Policy Lag Feature | 包含政策滞后特征</sup>',
        xaxis_title='Feature | 特征',
        yaxis_title='Importance Score | 重要性得分',
        barmode='group',
        template='plotly_white',
        height=400
    )
    
    return fig


# ============================================
# Main Application | 主应用
# ============================================

def main():
    # Title | 标题
    st.title("⚡ US Energy Forecast Pro | 美国能源消费预测")
    st.markdown("### Trump 2.0 Scenario Analysis | 情景分析 | XGBoost + CI + Policy Lag + Sensitivity")
    st.markdown("---")
    
    # ============================================
    # Sidebar - Scenario Settings | 侧边栏 - 情景设置
    # ============================================
    st.sidebar.header("🎛️ Scenario Settings | 情景参数")
    
    # Policy Parameters | 政策参数
    with st.sidebar.expander("📋 Policy Scores | 政策评分 (0-10)", expanded=True):
        green_subsidy = st.slider(
            "Green Subsidy Index | 绿色补贴指数",
            min_value=0, max_value=10, value=3,
            help="Expected Trump 2.0 to cut clean energy subsidies (2-year lag) | 预期削减清洁能源补贴（2年滞后生效）"
        )
        
        permitting_ease = st.slider(
            "Permitting Ease | 审批便利度",
            min_value=0, max_value=10, value=9,
            help="Deregulation for fossil projects | 化石能源项目放松管制"
        )
        
        trade_barrier = st.slider(
            "Trade Barrier | 贸易壁垒",
            min_value=0, max_value=10, value=9,
            help="Import tariffs | 进口关税"
        )
    
    # Macro Parameters | 宏观参数
    with st.sidebar.expander("📈 Macro Assumptions | 宏观经济假设 (%/yr)", expanded=True):
        gdp_growth = st.slider(
            "GDP Growth Rate | GDP增长率",
            min_value=-2.0, max_value=5.0, value=2.5, step=0.1
        )
        
        industrial_growth = st.slider(
            "Industrial Reshoring Growth | 工业回流增长率",
            min_value=-2.0, max_value=10.0, value=2.0, step=0.5
        )
        
        oil_price_change = st.slider(
            "Oil Price Change | 油价年变化率",
            min_value=-20.0, max_value=20.0, value=3.0, step=0.5
        )
    
    # Tech Parameters | 技术参数
    with st.sidebar.expander("🔧 Technology | 技术进步", expanded=False):
        lcoe_improvement = st.slider(
            "LCOE Improvement | LCOE年改善值 ($/MWh)",
            min_value=0.0, max_value=10.0, value=2.0, step=0.5,
            help="Annual renewable cost reduction | 可再生能源成本年度下降"
        )
    
    # Forecast Range | 预测范围
    st.sidebar.subheader("📅 Forecast Range | 预测范围")
    forecast_end = st.sidebar.selectbox(
        "Forecast Until | 预测至",
        options=[2026, 2027, 2028, 2029, 2030],
        index=2
    )
    
    # Build scenario params | 构建情景参数
    scenario_params = {
        'green_subsidy': green_subsidy,
        'permitting_ease': permitting_ease,
        'trade_barrier': trade_barrier,
        'gdp_growth_rate': gdp_growth,
        'industrial_growth_rate': industrial_growth,
        'oil_price_change': oil_price_change,
        'lcoe_improvement_per_year': lcoe_improvement
    }
    
    # ============================================
    # Data Loading | 数据加载
    # ============================================
    with st.spinner("Loading data... | 加载数据中..."):
        manual_df = load_manual_data()
        macro_df = fetch_fred_data(2000, 2024)
        merged_df = merge_all_data(manual_df, macro_df)
        df_with_lags = create_lag_features(merged_df)
    
    # Data Overview | 数据概览
    st.subheader("📊 Data Overview | 数据概览")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Data Years | 数据年份", f"{df_with_lags['Year'].min()}-{df_with_lags['Year'].max()}")
    with col2:
        st.metric("Observations | 观测数", len(df_with_lags))
    with col3:
        latest_fossil = df_with_lags['Fossil_Usage'].iloc[-1]
        st.metric("2024 Fossil | 化石能源", f"{latest_fossil:.1f} Q BTU")
    with col4:
        latest_renewable = df_with_lags['Renewable_Usage'].iloc[-1]
        st.metric("2024 Renewable | 可再生", f"{latest_renewable:.1f} Q BTU")
    
    st.markdown("---")
    
    # ============================================
    # Model Training | 模型训练
    # ============================================
    with st.spinner("Training XGBoost models... | 训练XGBoost模型..."):
        (fossil_model, renewable_model, fossil_features, renewable_features,
         fossil_rmse, renewable_rmse) = train_models(df_with_lags)
    
    # Model Accuracy | 模型精度
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"📊 Fossil Model RMSE | 化石模型: **{fossil_rmse:.3f}** Q BTU/yr")
    with col2:
        st.info(f"📊 Renewable Model RMSE | 可再生模型: **{renewable_rmse:.3f}** Q BTU/yr")
    
    # ============================================
    # Forecasting | 预测
    # ============================================
    last_row = df_with_lags.iloc[-1]
    forecast_years = list(range(2025, forecast_end + 1))
    
    with st.spinner("Generating forecast with CI... | 生成预测（含置信区间）..."):
        forecast_df = recursive_forecast(
            fossil_model, renewable_model,
            fossil_features, renewable_features,
            last_row, df_with_lags,
            scenario_params, forecast_years,
            fossil_rmse, renewable_rmse
        )
    
    # ============================================
    # Main Chart | 主图表
    # ============================================
    st.subheader("📈 Energy Trend & Forecast | 能源趋势与预测 (95% CI | 置信区间)")
    
    energy_display = st.radio(
        "Display Type | 显示类型",
        options=['both', 'fossil', 'renewable'],
        format_func=lambda x: {
            'both': 'All | 全部', 
            'fossil': 'Fossil Only | 仅化石能源', 
            'renewable': 'Renewable Only | 仅可再生能源'
        }[x],
        horizontal=True
    )
    
    fig_main = create_forecast_chart_with_ci(merged_df, forecast_df, energy_display)
    st.plotly_chart(fig_main, use_container_width=True)
    
    # Policy Lag Explanation | 政策滞后说明
    with st.expander("📌 Policy Lag Effect Explanation | 政策滞后效应说明"):
        st.markdown("""
        **Green_Subsidy_Lag2**: Green subsidy policies take ~**2 years** to impact actual energy consumption.
        
        **绿色补贴滞后2年**: 绿色补贴政策需要约**2年时间**才能影响实际能源消费。
        
        | Forecast Year | 预测年份 | Lag2 Source | 滞后来源 |
        |--------------|---------|-------------|---------|
        | 2025 | 2023 (Historical) | 历史数据 |
        | 2026 | 2024 (Historical) | 历史数据 |
        | 2027 | 2025 (Scenario) | 情景设定 |
        | 2028+ | Scenario Setting | 情景设定 |
        
        **Implication | 含义**: Even if Trump 2.0 cuts subsidies immediately, the impact on renewables won't fully materialize until **2 years later**.
        
        即使Trump 2.0立即削减补贴，对可再生能源的影响也要**2年后**才会完全显现。
        """)
    
    # ============================================
    # Forecast Results Table | 预测结果表格
    # ============================================
    st.subheader("📋 Forecast Details | 预测详情 (with 95% CI | 含置信区间)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Fossil Energy Forecast | 化石能源预测**")
        fossil_results = forecast_df[['Year', 'Fossil_Usage', 'Fossil_Lower', 'Fossil_Upper']].copy()
        fossil_results.columns = ['Year | 年份', 'Forecast | 预测', 'Lower | 下界', 'Upper | 上界']
        fossil_results = fossil_results.round(2)
        st.dataframe(fossil_results, use_container_width=True)
    
    with col2:
        st.markdown("**Renewable Energy Forecast | 可再生能源预测**")
        renewable_results = forecast_df[['Year', 'Renewable_Usage', 'Renewable_Lower', 'Renewable_Upper']].copy()
        renewable_results.columns = ['Year | 年份', 'Forecast | 预测', 'Lower | 下界', 'Upper | 上界']
        renewable_results = renewable_results.round(2)
        st.dataframe(renewable_results, use_container_width=True)
    
    # ============================================
    # Forecast Summary | 预测摘要
    # ============================================
    st.markdown("---")
    st.subheader("📊 Forecast Summary | 预测摘要")
    
    col1, col2, col3, col4 = st.columns(4)
    
    initial_fossil = merged_df[merged_df['Year'] == 2024]['Fossil_Usage'].values[0]
    final_fossil = forecast_df['Fossil_Usage'].iloc[-1]
    fossil_change = ((final_fossil - initial_fossil) / initial_fossil) * 100
    
    initial_renewable = merged_df[merged_df['Year'] == 2024]['Renewable_Usage'].values[0]
    final_renewable = forecast_df['Renewable_Usage'].iloc[-1]
    renewable_change = ((final_renewable - initial_renewable) / initial_renewable) * 100
    
    with col1:
        st.metric(
            f"{forecast_end} Fossil | 化石能源",
            f"{final_fossil:.2f} Q BTU",
            f"{fossil_change:+.1f}%"
        )
    
    with col2:
        st.metric(
            f"{forecast_end} Renewable | 可再生",
            f"{final_renewable:.2f} Q BTU",
            f"{renewable_change:+.1f}%"
        )
    
    with col3:
        total_2024 = initial_fossil + initial_renewable
        total_forecast = final_fossil + final_renewable
        total_change = ((total_forecast - total_2024) / total_2024) * 100
        st.metric(
            f"{forecast_end} Total | 总能源",
            f"{total_forecast:.2f} Q BTU",
            f"{total_change:+.1f}%"
        )
    
    with col4:
        renewable_share = (final_renewable / total_forecast) * 100
        st.metric(
            f"{forecast_end} Renewable Share | 可再生占比",
            f"{renewable_share:.1f}%"
        )
    
    # ============================================
    # Sensitivity Analysis Heatmap | 敏感性分析热力图
    # ============================================
    st.markdown("---")
    st.subheader("🔬 Policy vs Growth Sensitivity Analysis | 政策与增长敏感性分析")
    st.markdown("*Explore impact of different policy combinations on renewables | 探索不同政策组合对可再生能源的影响*")
    
    last_row_dict = last_row.to_dict()
    historical_subsidy = df_with_lags.set_index('Year')['Green_Subsidy_Index'].to_dict()
    
    with st.spinner("Calculating sensitivity matrix (121 simulations)... | 计算敏感性矩阵（121次模拟）..."):
        sensitivity_matrix = calculate_sensitivity(
            fossil_model, renewable_model,
            tuple(fossil_features), tuple(renewable_features),
            tuple(sorted(last_row_dict.items())),
            tuple(sorted(historical_subsidy.items())),
            scenario_params,
            target_year=2028
        )
    
    fig_heatmap, z_min, z_max, z_delta = create_sensitivity_heatmap(sensitivity_matrix, 2028)
    
    st.caption(
        f"🔬 **Microscope Mode | 显微镜模式** | Range | 范围: **{z_min:.4f}** → **{z_max:.4f}** Q BTU | "
        f"Delta | 变化幅度 (Δ): **{z_delta:.4f}** | Color scale optimized | 颜色比例已优化"
    )
    
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Sensitivity Insights | 敏感性洞察
    col1, col2, col3 = st.columns(3)
    with col1:
        max_val = sensitivity_matrix.max()
        max_idx = np.unravel_index(sensitivity_matrix.argmax(), sensitivity_matrix.shape)
        st.success(f"**Max Renewable | 最高可再生**: {max_val:.4f} Q BTU\n\nSubsidy | 补贴={max_idx[1]}, Growth | 增长={max_idx[0]}%")
    with col2:
        min_val = sensitivity_matrix.min()
        min_idx = np.unravel_index(sensitivity_matrix.argmin(), sensitivity_matrix.shape)
        st.error(f"**Min Renewable | 最低可再生**: {min_val:.4f} Q BTU\n\nSubsidy | 补贴={min_idx[1]}, Growth | 增长={min_idx[0]}%")
    with col3:
        current_val = sensitivity_matrix[int(industrial_growth), green_subsidy]
        st.info(f"**Current Scenario | 当前情景**: {current_val:.4f} Q BTU\n\nSubsidy | 补贴={green_subsidy}, Growth | 增长={industrial_growth}%")
    
    # ============================================
    # Feature Importance | 特征重要性
    # ============================================
    st.markdown("---")
    st.subheader("🔍 Feature Importance | 特征重要性")
    
    fig_importance = create_feature_importance_chart(
        fossil_model, renewable_model, fossil_features
    )
    st.plotly_chart(fig_importance, use_container_width=True)
    
    # ============================================
    # Data Panel | 数据面板
    # ============================================
    with st.expander("📁 View Full Data | 查看完整数据"):
        tab1, tab2, tab3 = st.tabs([
            "Historical | 历史数据", 
            "Forecast | 预测数据", 
            "Training | 训练数据"
        ])
        
        with tab1:
            st.dataframe(merged_df, use_container_width=True)
        with tab2:
            st.dataframe(forecast_df, use_container_width=True)
        with tab3:
            st.dataframe(df_with_lags, use_container_width=True)
    
    # ============================================
    # Methodology | 方法论
    # ============================================
    with st.expander("📖 Methodology | 方法论"):
        st.markdown("""
        ### Model Architecture | 模型架构
        
        **XGBoost + Difference Modeling | 差分建模**
        
        Based on XGBoost Recursive Forecasting with Lag features and Policy Sensitivity Analysis.
        
        基于 XGBoost 递归预测、滞后特征及政策敏感性分析。
        
        ---
        
        ### Pro Features | 专业版功能
        
        | Feature | 功能 | Description | 说明 |
        |---------|------|-------------|------|
        | 1️⃣ Uncertainty Quantification | 不确定性量化 | 95% Confidence Intervals using RMSE propagation | 使用RMSE传播的95%置信区间 |
        | 2️⃣ Policy Lag Effect | 政策滞后效应 | Green_Subsidy_Lag2 (2-year transmission) | 绿色补贴2年滞后传导 |
        | 3️⃣ Sensitivity Heatmap | 敏感性热力图 | 11×11 grid simulation (121 scenarios) | 11×11网格模拟 |
        | 4️⃣ Energy Intensity | 能源强度 | Captures efficiency trends, reduces OVB | 捕捉效率趋势，减少遗漏变量偏差 |
        
        ---
        
        ### Core Method | 核心方法
        
        - **Target | 目标**: `y = Energy_Diff` (Year-over-Year Change | 年度变化量)
        - **Solves | 解决**: Tree model extrapolation problem | 树模型外推问题
        - **Reconstruction | 重建**: `value(t) = value(t-1) + diff_pred(t)`
        
        ---
        
        ### Complete Feature List | 完整特征列表
        
        GDP, Industrial_Reshoring, Oil_Price, LCOE_Advantage, Green_Subsidy_Index, 
        Green_Subsidy_Lag2, Permitting_Ease, Trade_Barrier, Year_Index, 
        Energy_Intensity_Lag1, Fossil_Lag1 / Renewable_Lag1
        """)
    
    # Footer | 页脚
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "📊 US Energy Consumption Forecasting Pro | 美国能源消费预测 | "
        "XGBoost + CI + Policy Lag + Sensitivity | Trump 2.0 Scenario"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
