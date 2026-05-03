"""
Injaz Tech - AI Platform
Template for AI Tasks
=====================
Instructions:
1. Copy this file and rename it for your task (e.g., app_text_to_sql.py)
2. Update TASK_TITLE and TASK_ICON variables
3. Implement your task logic in the "TASK IMPLEMENTATION" section
4. Add your imports at the top
"""

import streamlit as st

# =============================================================================
# TASK CONFIGURATION - Modify these for your task
# =============================================================================
TASK_TITLE = "Your Task Title"  # e.g., "Text to SQL", "Text to Speech"
TASK_ICON = "🤖"                 # e.g., "🎤", "💬", "📊", "🗣️"

# =============================================================================
# Page Configuration
# =============================================================================
st.set_page_config(
    page_title=f"Injaz Tech | {TASK_TITLE}",
    page_icon=TASK_ICON,
    layout="centered",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# Custom CSS Styling - DO NOT MODIFY
# =============================================================================
st.markdown("""
<style>
    /* Main Colors */
    :root {
        --primary-color: #1A81A5;
        --secondary-color: #5AA7C0;
        --bg-color: #f8fafb;
        --card-bg: white;
        --text-dark: #333333;
        --text-light: #666666;
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, #1A81A5 0%, #5AA7C0 100%);
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 8px 24px rgba(26, 129, 165, 0.3);
    }
    
    .main-header h1 {
        color: white;
        font-size: 26px;
        margin: 0;
        font-weight: 600;
    }
    
    /* App Background */
    .stApp {
        background-color: #f8fafb;
    }
    
    /* Main Container Shadow */
    .block-container {
        background-color: white;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        margin-top: 1rem;
    }
    
    /* File Uploader Shadow */
    .stFileUploader > div {
        box-shadow: 0 2px 12px rgba(26, 129, 165, 0.15);
        border-radius: 10px;
    }
    
    /* Text Area Shadow */
    .stTextArea > div {
        box-shadow: 0 2px 12px rgba(26, 129, 165, 0.15);
        border-radius: 10px;
    }
    
    /* Select Box Shadow */
    .stSelectbox > div {
        box-shadow: 0 2px 12px rgba(26, 129, 165, 0.15);
        border-radius: 10px;
    }
    
    /* Text Input Shadow */
    .stTextInput > div {
        box-shadow: 0 2px 12px rgba(26, 129, 165, 0.15);
        border-radius: 10px;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #1A81A5 0%, #5AA7C0 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 30px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(26, 129, 165, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(26, 129, 165, 0.5);
    }
    
    /* Select Box Border */
    .stSelectbox > div > div {
        border-color: #5AA7C0;
    }
    
    /* Text Area Border */
    .stTextArea > div > div > textarea {
        border-color: #5AA7C0;
        border-radius: 8px;
    }
    
    /* Text Input Border */
    .stTextInput > div > div > input {
        border-color: #5AA7C0;
        border-radius: 8px;
    }
    
    /* Divider */
    .custom-divider {
        height: 3px;
        background: linear-gradient(90deg, #1A81A5, #5AA7C0, #1A81A5);
        border: none;
        border-radius: 2px;
        margin: 20px 0;
    }
    
    /* Success Box */
    .success-box {
        background-color: rgba(90, 167, 192, 0.2);
        border: 1px solid #5AA7C0;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(90, 167, 192, 0.25);
    }
    
    /* Info Box */
    .info-box {
        background-color: rgba(26, 129, 165, 0.1);
        border: 1px solid #1A81A5;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(26, 129, 165, 0.15);
    }
    
    /* Error Box */
    .error-box {
        background-color: rgba(220, 53, 69, 0.1);
        border: 1px solid #dc3545;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(220, 53, 69, 0.15);
    }
    
    /* Card Box */
    .card-box {
        background-color: white;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        margin: 10px 0;
    }
    
    /* Footer */
    .footer {
        background: linear-gradient(135deg, #1A81A5 0%, #5AA7C0 100%);
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin-top: 40px;
        box-shadow: 0 -4px 20px rgba(26, 129, 165, 0.2);
    }
    
    .footer p {
        color: white;
        margin: 0;
        font-size: 13px;
    }
    
    /* Section Title */
    .section-title {
        color: #1A81A5;
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 15px;
        padding-bottom: 8px;
        border-bottom: 2px solid #5AA7C0;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Logo (Top Right)
# =============================================================================
col1, col2 = st.columns([2, 1])
with col2:
    st.image("assets/injaz_tech.png", width=180)

# =============================================================================
# Header
# =============================================================================
st.markdown(f"""
<div class="main-header">
    <h1>{TASK_ICON} {TASK_TITLE}</h1>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# TASK IMPLEMENTATION - Add your task logic below
# =============================================================================
 ## Your code goes here ##

# =============================================================================
# Footer - DO NOT MODIFY
# =============================================================================
st.markdown("""
<div class="footer">
    <p>© 2025 Injaz Tech</p>
</div>
""", unsafe_allow_html=True)
    
