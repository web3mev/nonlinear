import streamlit as st
import pandas as pd
import polars as pl
import numpy as np
import nonlinear_fitting_numba as nlf
import ui_layer as ui # Import new UI layer
import os

import data_manager as dm

st.set_page_config(page_title="Fitting", layout="wide")

# Inject CSS
st.markdown("""
    <style>

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

# Load Data via Data Manager
df_loaded = dm.load_and_process_data(config)
if df_loaded is not None:
    st.session_state.fitting_data = df_loaded


# --- 3. UI Sections (Tabs) ---
# --- 3. UI Sections (Tabs) ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Data Exploration", "Model Specification", "Model Fitting", "Advanced", "Machine Learning Fitting", "Neural Network Fitting", "Documentation"])

with tab1:
    ui.render_exploration()

with tab2:
    ui.render_input_section(config['param_file'])

with tab3:
    ui.render_fitting_control(config)
    st.markdown("---")
    ui.render_results()

with tab4:
    ui.render_advanced_features(config)

with tab5:
    ui.render_ml_fitting(config)

with tab6:
    ui.render_nn_fitting(config)

with tab7:
    ui.render_documentation()

