"""
数据处理脚本 - process_data.py
合并用户原始CSV与专家逻辑数据（LCOE & 政策评分）
"""

import pandas as pd
import numpy as np
import os

# ============================================
# 专家逻辑数据 - 硬编码字典 (2000-2024)
# ============================================

# LCOE优势 (天然气成本 - 太阳能成本, $/MWh)
LCOE_ADVANTAGE = {
    2000: -350, 2001: -340, 2002: -330, 2003: -300, 2004: -290,
    2005: -280, 2006: -280, 2007: -280, 2008: -270, 2009: -276,
    2010: -166, 2011: -86, 2012: -53, 2013: -34, 2014: -5,
    2015: 1, 2016: 8, 2017: 10, 2018: 15, 2019: 15,
    2020: 22, 2021: 24, 2022: 17, 2023: 10, 2024: 15
}

# 绿色补贴指数 (政策评分 1-10)
GREEN_SUBSIDY_INDEX = {
    2000: 2, 2001: 2, 2002: 2, 2003: 2, 2004: 2,
    2005: 2, 2006: 3, 2007: 3, 2008: 3, 2009: 6,
    2010: 6, 2011: 6, 2012: 6, 2013: 6, 2014: 6,
    2015: 6, 2016: 6, 2017: 3, 2018: 3, 2019: 3,
    2020: 3, 2021: 9, 2022: 9, 2023: 9, 2024: 9
}

# 审批便利度 (放松管制评分 1-10)
PERMITTING_EASE = {
    2000: 6, 2001: 6, 2002: 6, 2003: 6, 2004: 6,
    2005: 6, 2006: 6, 2007: 6, 2008: 6, 2009: 4,
    2010: 4, 2011: 4, 2012: 4, 2013: 4, 2014: 4,
    2015: 4, 2016: 4, 2017: 8, 2018: 8, 2019: 8,
    2020: 8, 2021: 3, 2022: 3, 2023: 3, 2024: 3
}

# 贸易壁垒 (关税评分 1-10)
TRADE_BARRIER = {
    2000: 1, 2001: 1, 2002: 1, 2003: 1, 2004: 1,
    2005: 1, 2006: 1, 2007: 1, 2008: 1, 2009: 1,
    2010: 1, 2011: 1, 2012: 3, 2013: 3, 2014: 3,
    2015: 3, 2016: 3, 2017: 3, 2018: 8, 2019: 8,
    2020: 8, 2021: 9, 2022: 9, 2023: 9, 2024: 9
}


def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    加载用户原始CSV文件
    
    Args:
        filepath: CSV文件路径
        
    Returns:
        pandas DataFrame
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"找不到文件: {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"✓ 成功加载原始数据: {filepath}")
    print(f"  - 数据形状: {df.shape}")
    print(f"  - 列名: {list(df.columns)}")
    return df


def inject_expert_logic(df: pd.DataFrame) -> pd.DataFrame:
    """
    注入专家逻辑数据 - 如果CSV缺少技术/政策列，则使用硬编码数据填充
    
    Args:
        df: 原始DataFrame
        
    Returns:
        包含所有特征的DataFrame
    """
    df = df.copy()
    
    # 确保Year列存在
    if 'Year' not in df.columns:
        raise ValueError("CSV必须包含'Year'列")
    
    # 定义需要注入的列及其对应数据
    expert_data = {
        'LCOE_Advantage': LCOE_ADVANTAGE,
        'Green_Subsidy_Index': GREEN_SUBSIDY_INDEX,
        'Permitting_Ease': PERMITTING_EASE,
        'Trade_Barrier': TRADE_BARRIER
    }
    
    injected_cols = []
    existing_cols = []
    
    for col_name, data_dict in expert_data.items():
        if col_name not in df.columns:
            # 列不存在，注入数据
            df[col_name] = df['Year'].map(data_dict)
            injected_cols.append(col_name)
        else:
            existing_cols.append(col_name)
    
    if injected_cols:
        print(f"✓ 已注入专家逻辑数据列: {injected_cols}")
    if existing_cols:
        print(f"ℹ 已存在的列(保留原值): {existing_cols}")
    
    return df


def validate_data(df: pd.DataFrame) -> bool:
    """
    验证数据完整性
    
    Args:
        df: DataFrame
        
    Returns:
        验证是否通过
    """
    required_cols = ['Year', 'Fossil_Usage', 'Renewable_Usage', 
                     'LCOE_Advantage', 'Green_Subsidy_Index', 
                     'Permitting_Ease', 'Trade_Barrier']
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"✗ 缺少必需列: {missing_cols}")
        return False
    
    # 检查缺失值
    null_counts = df[required_cols].isnull().sum()
    if null_counts.sum() > 0:
        print("⚠ 发现缺失值:")
        for col, count in null_counts.items():
            if count > 0:
                print(f"  - {col}: {count}个缺失值")
        
        # 使用线性插值填充缺失值
        df[required_cols] = df[required_cols].interpolate(method='linear')
        print("✓ 已使用线性插值填充缺失值")
    
    print("✓ 数据验证通过")
    return True


def save_processed_data(df: pd.DataFrame, output_path: str):
    """
    保存处理后的数据
    
    Args:
        df: DataFrame
        output_path: 输出文件路径
    """
    df.to_csv(output_path, index=False)
    print(f"✓ 已保存处理后的数据到: {output_path}")
    print(f"  - 数据形状: {df.shape}")
    print(f"  - 年份范围: {df['Year'].min()} - {df['Year'].max()}")


def main():
    """主函数"""
    print("=" * 60)
    print("数据处理脚本 - US Energy Consumption Forecasting")
    print("=" * 60)
    
    # 文件路径
    input_file = "prediction_data.csv"
    output_file = "manual_data.csv"
    
    try:
        # Step 1: 加载原始数据
        print("\n[Step 1] 加载原始数据...")
        df = load_raw_data(input_file)
        
        # Step 2: 注入专家逻辑
        print("\n[Step 2] 注入专家逻辑数据...")
        df = inject_expert_logic(df)
        
        # Step 3: 验证数据
        print("\n[Step 3] 验证数据完整性...")
        validate_data(df)
        
        # Step 4: 保存处理后的数据
        print("\n[Step 4] 保存处理后的数据...")
        save_processed_data(df, output_file)
        
        print("\n" + "=" * 60)
        print("✓ 数据处理完成!")
        print("=" * 60)
        
        # 显示数据预览
        print("\n数据预览:")
        print(df.head(10).to_string())
        
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        raise


if __name__ == "__main__":
    main()
