import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nonlinear_fitting_numba as nlf
import time
import os
import glob

st.set_page_config(page_title="Nonlinear Fitting Tool", layout="wide")

def get_next_version_filename(base_name="parameter_fitted"):
    files = glob.glob(f"{base_name}_v*.csv")
    if not files:
        return f"{base_name}_v1.csv"
    
    max_v = 0
    for f in files:
        try:
            # Extract version number: parameter_fitted_v1.csv -> 1
            v_str = f.split("_v")[-1].replace(".csv", "")
            v = int(v_str)
            if v > max_v:
                max_v = v
        except:
            pass
    return f"{base_name}_v{max_v + 1}.csv"

st.title("Nonlinear Fitting Tool")

# --- Sidebar ---
st.sidebar.header("Configuration")

uploaded_param_file = st.sidebar.file_uploader("Upload Parameter File", type=["csv"])
if uploaded_param_file:
    param_file = uploaded_param_file
else:
    param_file = "parameters.csv"

if st.sidebar.button("Reload Parameters"):
    st.cache_data.clear()
    if 'params_df' in st.session_state:
        del st.session_state.params_df
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.header("Data Configuration")

# Data Source Selection Logic
data_source = st.sidebar.radio("Data Source", ["Generate Data", "Load Data File"])

sample_size = st.sidebar.number_input("Sample Size", min_value=1000, value=100000, step=1000)

data_file = None
if data_source == "Load Data File":
    data_file = st.sidebar.file_uploader("Upload Data File", type=["csv", "parquet"])

# --- Main Content ---

# 1. Load Parameters
@st.cache_data
def load_params(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

if 'params_df' not in st.session_state:
    df = load_params(param_file)
    if df is not None:
        st.session_state.params_df = df

if 'params_df' in st.session_state:
    # Streamlit 1.51.0 supports data_editor for direct editing.
    # We use st.data_editor to allow users to edit the dataframe directly.
    
    st.markdown("### Edit Parameters")
    st.markdown("You can edit values directly in the table below.")
    
    # Configure column settings for better UX
    # We enable all columns to support full table copy/paste from Excel.
    column_config = {
        "RiskFactor_NM": st.column_config.TextColumn("Risk Factor"),
        "Sub_Model": st.column_config.TextColumn("Sub Model"),
        "RiskFactor": st.column_config.NumberColumn("Value", format="%.4f"),
        "On_Off": st.column_config.SelectboxColumn("On/Off", options=["Y", "N"]),
        "Fixed": st.column_config.SelectboxColumn("Fixed", options=["Y", "N"]),
        "Monotonicity": st.column_config.TextColumn("Monotonicity"), # Keep as text to allow "nan" or numbers
    }

    edited_df = st.data_editor(
        st.session_state.params_df,
        column_config=column_config,
        # num_rows="dynamic", # Removed to improve copy-paste stability for existing rows
        use_container_width=True, # Keeping this for now despite warning to ensure layout stability
        height=400,
        key="data_editor"
    )
    
    # Update session state with edits
    st.session_state.params_df = edited_df

    # Update session state with edits
    st.session_state.params_df = edited_df

    st.info("💡 Tip: You can copy and paste cells directly within the table (Ctrl+C / Ctrl+V).")

    st.markdown("---")
    if st.button("Save Parameters to CSV"):
        try:
            st.session_state.params_df.to_csv(param_file, index=False)
            st.success(f"Saved to {param_file}")
        except Exception as e:
            st.error(f"Error saving file: {e}")

    # 2. Fitting Control
    st.markdown("### Fitting Control")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        start_btn = st.button("Start Fitting")
    
    if start_btn:
        st.info("Starting optimization process...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(p, text):
            progress_bar.progress(p)
            status_text.text(text)
            
        # Data Handling Logic
        df_data = None
        true_values = None
        
        try:
            # Load Model Spec first to get components for generation
            components = nlf.load_model_spec(df=edited_df)
            
            if data_source == "Load Data File" and data_file:
                update_progress(0.1, f"Loading data from {data_file.name}...")
                try:
                    if data_file.name.endswith(".parquet"):
                        df_loaded = pd.read_parquet(data_file)
                    else:
                        df_loaded = pd.read_csv(data_file)
                    
                    # Sampling
                    if len(df_loaded) > sample_size:
                        update_progress(0.15, f"Sampling {sample_size} rows from {len(df_loaded)}...")
                        df_data = df_loaded.sample(n=sample_size)
                    else:
                        df_data = df_loaded
                    
                    # Weight Calculation
                    if 'balance' in df_data.columns:
                        # w = 1 / sqrt(balance)
                        # Handle zeros/negatives by clipping to small epsilon
                        balance = df_data['balance'].clip(lower=1e-6)
                        df_data['w'] = 1.0 / np.sqrt(balance)
                        update_progress(0.18, "Calculated weights from 'balance' column.")
                    elif 'w' not in df_data.columns:
                        df_data['w'] = 1.0
                        update_progress(0.18, "No 'balance' or 'w' column found. Using default weights (1.0).")
                        
                except Exception as e:
                    st.error(f"Failed to load data file: {e}")
                    st.stop()
            else:
                # Generate Data
                update_progress(0.1, f"Generating {sample_size} rows of synthetic data...")
                df_data, true_values = nlf.generate_data(components, n_samples=sample_size)
                
                # Save generated data to parquet
                save_path = "generated_data.parquet"
                df_data.to_parquet(save_path)
                st.info(f"Generated data saved to {os.path.abspath(save_path)}")

            # Run Fitting with prepared data
            results = nlf.run_fitting_api(
                df_params=edited_df,
                df_data=df_data,
                true_values=true_values,
                progress_callback=update_progress
            )
            
            st.session_state.fitting_results = results
            st.success("Fitting Completed!")
            
        except Exception as e:
            st.error(f"An error occurred during fitting: {e}")
            raise e

    # 3. Results Display
    if 'fitting_results' in st.session_state:
        results = st.session_state.fitting_results
        
        st.markdown("---")
        st.header("Results")
        
        # Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Success", str(results['success']))
        m2.metric("Final Cost", f"{results['cost']:.4f}")
        m3.metric("Time Elapsed", f"{results['time']:.2f} s")
        
        if 'metrics' in results:
            mets = results['metrics']
            m4, m5, m6, m7 = st.columns(4)
            m4.metric("R-squared", f"{mets['R2']:.4f}")
            m5.metric("RMSE", f"{mets['RMSE']:.4f}")
            m6.metric("MAE", f"{mets['MAE']:.4f}")
            m7.metric("N Samples", f"{mets['n_samples']:,}")
        
        # Fitted Parameters (Compact)
        with st.expander("Fitted Parameters Table"):
            display_df = results['fitted_params'].copy()
            display_df['X1_Knot'] = display_df['X1_Knot'].astype(str)
            display_df['X2_Knot'] = display_df['X2_Knot'].astype(str)
            st.dataframe(display_df)
        
        # Plots
        st.subheader("Diagnostic Plots")
        figures = results['figures']
        
        # Summary Plots
        # Summary Plots
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.pyplot(figures['Actual vs Predicted'])
        with c2:
            st.pyplot(figures['Residuals vs Predicted'])
        with c3:
            st.pyplot(figures['Histogram of Residuals'])
        with c4:
            st.pyplot(figures['Q-Q Plot'])
        
        # Performance Charts (New Section)
        st.subheader("Performance Charts (Weighted Actual vs Model)")
        perf_figs = {k: v for k, v in figures.items() if k.startswith("Performance:")}
        if perf_figs:
            # Group by parameter name
            grouped_figs = {}
            for k, v in perf_figs.items():
                # Extract param name. 
                # Format: "Performance: {comp['name']} (By {comp['x1_var']})" or "Performance: {comp['name']}"
                clean_k = k.replace("Performance: ", "")
                if " (By " in clean_k:
                    param_name = clean_k.split(" (By ")[0]
                else:
                    param_name = clean_k
                
                if param_name not in grouped_figs:
                    grouped_figs[param_name] = []
                grouped_figs[param_name].append((k, v))
            
            # Buffer for 1D plots
            buffer_1d = []
            
            def flush_buffer_1d(buf):
                if not buf: return
                cols = st.columns(4)
                for i, (name, fig) in enumerate(buf):
                    with cols[i % 4]:
                        st.pyplot(fig)
                buf.clear()

            for param_name, figs_list in grouped_figs.items():
                if len(figs_list) == 2:
                    # 2D Parameter - Flush buffer first
                    flush_buffer_1d(buffer_1d)
                    
                    st.markdown(f"##### {param_name}")
                    cols = st.columns(2)
                    for i, (name, fig) in enumerate(figs_list):
                        with cols[i]:
                            st.pyplot(fig)
                else:
                    # 1D Parameter (or other count) - Add to buffer
                    buffer_1d.extend(figs_list)
            
            # Final flush
            flush_buffer_1d(buffer_1d)
        
        # Component Plots (Fitted Curves/Surfaces)
        st.subheader("Component Plots")
        comp_figs = {k: v for k, v in figures.items() if k.startswith("Component:")}
        cols = st.columns(4)
        for i, (name, fig) in enumerate(comp_figs.items()):
            with cols[i % 4]:
                st.pyplot(fig)
            
        st.markdown("---")
        if st.button("Export Fitted Parameters"):
            export_df = st.session_state.params_df.copy()
            fitted_df = st.session_state.fitting_results['fitted_params']
            # Merge fitted values back into export_df
            # We match on RiskFactor_NM, X1_Val, X2_Val
            
            # Create helper columns for robust matching
            # We'll create a unique key: "{RiskFactor_NM}|{X1_Val}|{X2_Val}"
            # We need to handle float precision and nan consistently
            
            def create_key(row, nm_col, x1_col, x2_col):
                nm = str(row[nm_col])
                
                # Helper to format value
                def fmt(val):
                    if pd.isna(val) or str(val).strip() in ['-', 'nan', 'None']:
                        return "0" # Treat missing/dash as 0 for matching
                    try:
                        f = float(val)
                        if f.is_integer():
                            return str(int(f))
                        return f"{f:.6f}" # Use fixed precision
                    except:
                        return str(val)
                        
                x1 = fmt(row[x1_col])
                x2 = fmt(row[x2_col])
                return f"{nm}|{x1}|{x2}"

            # 1. Create keys for export_df (Target)
            export_df['match_key'] = export_df.apply(
                lambda r: create_key(r, 'RiskFactor_NM', 'X1_Val', 'X2_Val'), axis=1
            )
            
            # 2. Create keys for fitted_df (Source)
            fitted_df['match_key'] = fitted_df.apply(
                lambda r: create_key(r, 'Parameter', 'X1_Knot', 'X2_Knot'), axis=1
            )
            
            # 3. Map fitted values
            # Create a dictionary mapping key -> fitted_value
            fit_map = pd.Series(
                fitted_df['Fitted_Value'].values,
                index=fitted_df['match_key']
            ).to_dict()
            
            # 4. Update RiskFactor where key matches
            # We iterate or use map. Map is faster.
            
            # Identify rows that have a match
            mask = export_df['match_key'].isin(fit_map.keys())
            
            # Update values
            export_df.loc[mask, 'RiskFactor'] = export_df.loc[mask, 'match_key'].map(fit_map)
            
            # Drop helper column
            export_df.drop(columns=['match_key'], inplace=True)
            
            # Save to new versioned file
            fname = get_next_version_filename()
            try:
                export_df.to_csv(fname, index=False)
                abs_path = os.path.abspath(fname)
                st.success(f"Successfully exported fitted parameters to **{abs_path}**")
            except Exception as e:
                st.error(f"Error exporting file: {e}")
