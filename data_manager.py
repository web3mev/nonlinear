import streamlit as st
import polars as pl
import pandas as pd

def load_and_process_data(config):
    """
    Loads data from file, applies sampling ratio, and calculates weights.
    Returns: Dataframe (with 'w', 'balance' columns) or None
    """
    if config['data_source'] == "Load Data File" and config['data_file']:
        try:
            # 1. Load File
            if config['data_file'].name.endswith(".parquet"):
                df_loaded = pl.read_parquet(config['data_file'])
            else:
                df_loaded = pl.read_csv(config['data_file'])

            # 2. Sampling
            sampling_ratio = config.get('sampling_ratio', 1.0)
            if sampling_ratio < 1.0:
                print(f"Sampling {sampling_ratio*100}% of loaded data...")
                df_data = df_loaded.sample(fraction=sampling_ratio, seed=42)
            else:
                df_data = df_loaded
            
            # 3. Weighting
            if config['ignore_weights']:
                df_data = df_data.with_columns(pl.lit(1.0).alias('w'))
            elif 'balance' in df_data.columns:
                t = config.get('balance_power_t', 1.0)
                # w = balance^t
                df_data = df_data.with_columns(
                    (pl.col('balance').clip(1e-6, None).pow(t)).alias('w')
                )
            elif 'weight' in df_data.columns:
                df_data = df_data.with_columns(pl.col('weight').alias('w'))
            elif 'w' in df_data.columns:
                pass # Already has w
            else:
                df_data = df_data.with_columns(pl.lit(1.0).alias('w'))
            
            return df_data
            
        except Exception as e:
            st.error(f"Error processing data: {e}")
            return None
    return None
