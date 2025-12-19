import streamlit as st

def apply_tight_layout():
    """Injects CSS to reduce padding and margins for a tighter layout."""
    st.markdown("""
        <style>
               /* Global padding reduction for main container */
               .block-container {
                    padding-top: 1rem !important;
                    padding-bottom: 1rem !important;
                    padding-left: 2rem !important;
                    padding-right: 2rem !important;
               }
               
               /* Reduce vertical spacing between elements */
               .element-container, .stMarkdown, .stButton {
                    margin-bottom: 0.2rem !important;
               }
               
               /* Reduce sidebar padding */
               section[data-testid="stSidebar"] .block-container {
                    padding-top: 1rem !important;
                    padding-left: 1rem !important;
                    padding-right: 1rem !important;
               }
               
               /* Compact headers */
               h1, h2, h3, h4, h5 {
                    margin-top: 0.5rem !important;
                    margin-bottom: 0.5rem !important;
                    padding-bottom: 0px !important;
               }
               
               /* Expander compactness */
               .streamlit-expanderHeader {
                    padding-top: 0.2rem !important;
                    padding-bottom: 0.2rem !important;
               }
               
               /* Divider removal just in case */
               hr {
                    margin-top: 0.5rem !important;
                    margin-bottom: 0.5rem !important;
               }
        </style>
    """, unsafe_allow_html=True)
