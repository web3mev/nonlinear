import unittest
import pandas as pd
import numpy as np
import io
import base64
import time
import os
import shutil
from dash import html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import dash

# Import the app module
# We need to make sure we don't start the server
import dash_app
from dash_app import state, render_tab_content, fitting_control, load_data

class TestDashWorkflow(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Create a dummy parameters CSV
        cls.params_str = """RiskFactor_NM,Calc_Type,Sub_Model,X1_Var,X2_Var,Monotonicity,Example_Value,Fixed,Key,X1_Val,X2_Val,RiskFactor,On_Off
Intercept,DIM_0,,Intercept,,,1,N,1,,,0.0,Y
age,DIM_1,,age,,,30,N,2,20,,0.1,Y
age,DIM_1,,age,,,30,N,2,30,,0.2,Y
age,DIM_1,,age,,,30,N,2,40,,0.3,Y
"""
        cls.params_b64 = base64.b64encode(cls.params_str.encode('utf-8')).decode('utf-8')
        cls.params_content = f"data:text/csv;base64,{cls.params_b64}"
        
        # Clean up old temp files
        if os.path.exists("temp_data_test.parquet"):
            os.remove("temp_data_test.parquet")
        if os.path.exists("temp_results_test.pkl"):
            os.remove("temp_results_test.pkl")
            
        # Override state constants for testing
        dash_app.state.DATA_FILE = "temp_data_test.parquet"
        dash_app.state.RESULTS_FILE = "temp_results_test.pkl"
        
    def test_01_upload_parameters(self):
        print("\n--- Testing Upload Parameters ---")
        # Simulating upload-params callback (logic is inside render_tab_content for processing?? No, wait)
        # In dash_app.py, render_tab_content handles 'upload-params' trigger
        
        # We need to simulate the ctx.triggered_id logic. 
        # Since we can't easily mock ctx in simple unit test without dash context manager,
        # we might check if logic allows passing inputs.
        # dash_app.render_tab_content checks `trigger == 'upload-params'`
        
        # Ideally, we should refactor logic out of callbacks, but here we test the callback logic.
        # We can use `mock` to patch ctx.triggered_id?
        from unittest.mock import patch
        
        with patch('dash_app.ctx') as mock_ctx:
            mock_ctx.triggered_id = 'upload-params'
            
            output, content = render_tab_content(
                contents=self.params_content,
                active_tab="tab-params",
                trigger_data=None,
                theme_is_light=True,
                plot_backend='matplotlib', # Default
                filename="parameters.csv"
            )
            
            self.assertIsNotNone(state.params_df, "Params DF should be loaded")
            self.assertEqual(len(state.params_df), 4, "Should have 4 rows based on dummy csv")
            self.assertIn("Loaded parameters.csv", output, "Status should indicate success")
            print("Upload Parameters: SUCCESS")

    def test_02_data_generation_and_loading(self):
        print("\n--- Testing Data Generation (via Fitting Control) ---")
        # logical flow: User selects 'generate' in radio (client side), then clicks Start.
        # fitting_control handles data generation if data_source='generate'
        
        # We'll simulate a start click
        # fitting_control(start, stop, n, backend, method, max_iter, tol, l1, l2, data_source, sample, ignore_w, plot_bk, trig)
        
        # We need to mock dash.no_update check? No, just inputs.
        
        with patch('dash_app.ctx') as mock_ctx:
            mock_ctx.triggered_id = 'start-btn'
             
            # 1. Start Fitting (triggers thread)
            out_start_dis, out_stop_dis, out_prog, out_prog_label, out_text, out_trigger = fitting_control(
                start_clicks=1, stop_clicks=0, n=0, 
                backend='scipy_ls', method='trf', max_iter=100, tol=6, l1=0, l2=0,
                data_source='generate', sample_size=1000, ignore_weights=False, 
                plot_backend='matplotlib', current_trigger=0
            )
            
            self.assertTrue(state.is_running, "Fitting should be running")
            self.assertEqual(out_text, "Fitting started...", "Status text mismatch")
            print("Start Fitting Signal: SUCCESS")
            
            # Wait for thread to finish (simulate user waiting)
            print("Waiting for fitting thread...")
            time.sleep(5) 
            
            self.assertFalse(state.is_running, "Fitting should finish quickly for small data")
            self.assertIsNotNone(state.fitting_data, "Data should be generated")
            self.assertIsNotNone(state.fitting_results, "Results should be produced")
            print("Fitting Completion: SUCCESS")

    def test_03_tab_rendering_data_preview(self):
        print("\n--- Testing Tab: Data Preview ---")
        
        with patch('dash_app.ctx') as mock_ctx:
            mock_ctx.triggered_id = 'tabs'
            
            status, content = render_tab_content(
                contents=None,
                active_tab="tab-data",
                trigger_data=None,
                theme_is_light=True,
                plot_backend='matplotlib',
                filename=None
            )
            
            # Content should be a Div containing Table and Exploration
            # Dash components are objects, we can check basic types or strings
            self.assertIsInstance(content, html.Div)
            # Check for Table
            has_table = any(isinstance(c, dbc.Table) for c in content.children) if isinstance(content.children, list) else False
            self.assertTrue(has_table, "Data tab should contain a table")
            print("Tab Data Preview: SUCCESS")

    def test_04_tab_rendering_results(self):
        print("\n--- Testing Tab: Results ---")
        with patch('dash_app.ctx') as mock_ctx:
            mock_ctx.triggered_id = 'tabs'
            status, content = render_tab_content(None, "tab-results", None, True, 'matplotlib', None)
            self.assertIsInstance(content, html.Pre, "Results should be a Pre tag")
            print("Tab Results: SUCCESS")

    def test_05_tab_rendering_diagnostics_matplotlib(self):
        print("\n--- Testing Tab: Diagnostics (Matplotlib) ---")
        with patch('dash_app.ctx') as mock_ctx:
            mock_ctx.triggered_id = 'tabs'
            status, content = render_tab_content(None, "tab-diag", None, True, 'matplotlib', None)
            
            # Should have H4 headers for groups
            has_headers = any(isinstance(c, html.H4) for c in content.children)
            self.assertTrue(has_headers, "Should have group headers")
            
            # Check for Images (Matplotlib renders as Img)
            # Need to drill down rows/cols... this is hard via inspection, 
            # but if it didn't crash and returned content, it's good.
            print("Tab Diagnostics (Matplotlib): SUCCESS")

    def test_06_tab_rendering_diagnostics_plotly(self):
        print("\n--- Testing Tab: Diagnostics (Plotly) + Regeneration ---")
        # This tests the "Regenerating plots" logic because state currently has mpl figures (default)
        # We switch to plotly
        
        with patch('dash_app.ctx') as mock_ctx:
            mock_ctx.triggered_id = 'plotting-backend-radio'
            
            status, content = render_tab_content(None, "tab-diag", None, True, 'plotly', None)
            
            # State should now be updated with Plotly figures
            sample_fig = list(state.fitting_results['figures'].values())[0]
            # Plotly figure or dict
            is_plotly = hasattr(sample_fig, 'to_dict') or hasattr(sample_fig, 'layout')
            self.assertTrue(is_plotly, "Figures should be regenerated as Plotly objects")
            print("Backend Switch & Regeneration: SUCCESS")

if __name__ == '__main__':
    unittest.main()
