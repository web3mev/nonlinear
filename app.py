import streamlit as st
import pandas as pd
import polars as pl
import numpy as np
import nonlinear_fitting_numba as nlf
import ui_layer as ui # Import new UI layer
import os

st.set_page_config(page_title="Fitting", layout="wide")

# Inject CSS
st.markdown("""
    <style>
        header {visibility: hidden;}
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
        }
        input, textarea {
            caret-color: black;
        }
        @media (prefers-reduced-motion: no-preference) {
            * {
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                scroll-behavior: auto !important;
            }
        }
    </style>
""", unsafe_allow_html=True)

st.title("Fitting")

# --- 1. Sidebar & Configuration ---
start_time = os.times()
config = ui.render_sidebar()

# --- 2. Data Loading & State Initialization ---
# Initialize Session Keys
if 'fitting_data' not in st.session_state: st.session_state.fitting_data = None
if 'params_df' not in st.session_state:
    try:
        # Initial load
        if hasattr(config['param_file'], 'read'): # Uploaded file
             st.session_state.params_df = pd.read_csv(config['param_file'])
        else: # String path
             st.session_state.params_df = pd.read_csv(config['param_file'])
    except Exception as e:
        st.error(f"Error loading parameters: {e}")

# Load Data logic (Keep minimal in main or move to helper?)
# Keeping some logic here to bridge config and state
if config['data_source'] == "Load Data File" and config['data_file']:
    @st.cache_data
    def load_data_file(uploaded_file):
        try:
            if uploaded_file.name.endswith(".parquet"): return pl.read_parquet(uploaded_file)
            else: return pl.read_csv(uploaded_file)
        except: return None

    df_loaded = load_data_file(config['data_file'])
    if df_loaded is not None:
        # Sampling & Weight check
        try:
            if len(df_loaded) > config['sample_size']: df_data = df_loaded.sample(n=config['sample_size'], seed=42)
            else: df_data = df_loaded
            
            if config['ignore_weights']: df_data = df_data.with_columns(pl.lit(1.0).alias('w'))
            elif 'weight' in df_data.columns: df_data = df_data.with_columns(pl.col('weight').alias('w'))
            elif 'w' in df_data.columns: pass
            elif 'balance' in df_data.columns:
                t = config.get('balance_power_t', 0.0)
                # w = balance^t
                df_data = df_data.with_columns(
                    (pl.col('balance').clip(1e-6, None).pow(t)).alias('w')
                )
            else: df_data = df_data.with_columns(pl.lit(1.0).alias('w'))
            
            st.session_state.fitting_data = df_data
            # Clear old results if data changes?
            # if 'exploration_results' in st.session_state: st.session_state.exploration_results = None
        except Exception as e:
            st.error(f"Error processing data: {e}")

# --- 3. UI Sections ---
ui.render_input_section(config['param_file'])
ui.render_exploration()
ui.render_fitting_control(config)
ui.render_results()

# --- 4. Advanced Features (Bootstrap, CV, MCMC) ---
# These were not yet moved to ui_layer in the previous step,
# but to keep app.py clean I should probably move them or keep them compact.
# For Phase 1, I can keep them or verify if I missed adding them to ui_layer.
# I missed adding them to ui_layer. I will add them to ui_layer in a subsequent step or append them now.
# Given "Implement All", I should move them. 
# But let's first get the main app working with what we have, then append the advanced ones.
# Actually, strict refactoring suggests I should have added them.
# I will append the advanced features logic to ui_layer in the NEXT tool call, 
# then here I will just call `ui.render_advanced_features(config)`.

ui.render_advanced_features(config)
