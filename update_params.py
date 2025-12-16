import pandas as pd
import os

try:
    df = pd.read_csv('parameters.csv')
    rename_map = {
        'X1_Var': 'X1_Var_NM',
        'X2_Var': 'X2_Var_NM',
        'RiskFactor': 'RiskFactor_VAL',
        'X1_Val': 'X1_Var_Val',
        'X2_Val': 'X2_Var_Val',
        'On_Off': 'On_Off_Flag'
    }
    df.rename(columns=rename_map, inplace=True)
    if 'Notes' not in df.columns:
        df['Notes'] = ""
    
    df.to_csv('parameters.csv', index=False)
    print("Successfully updated parameters.csv")
except Exception as e:
    print(f"Error: {e}")
