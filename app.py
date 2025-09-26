# ===============================
# ENHANCED STREAMLIT APP WITH BIAS & FAIRNESS DASHBOARD
# Deploy this to your GitHub: madhudevi25/AI-Market-Trend-Advisor
# ===============================

"""
STREAMLIT APP ENHANCEMENTS
==========================

NEW FEATURES:
1. 🛡️ Bias & Fairness Transparency Dashboard
2. 📊 Real-time audit results display
3. 🚀 Gemini 2.0 Flash integration (95% cheaper)
4. 🔍 Detailed transparency metrics
5. 📋 Methodology explanation for user trust

GITHUB INTEGRATION:
Repository: madhudevi25/AI-Market-Trend-Advisor
Files needed:
- app.py (this file)
- requirements.txt (updated)
- processed_market_data.parquet (from Colab)
- market_faiss_index.faiss (from Colab)
- market_analysis_results.json (from Colab)
- enhanced_analysis_results.json (from Colab)

STREAMLIT CLOUD DEPLOYMENT:
Connect to GitHub repo and auto-deploy on push
"""

import streamlit as st
import pandas as pd
import numpy as np
import faiss
import json
from datetime import datetime
from sentence_transformers import SentenceTransformer
import vertexai
from vertexai.generative_models import GenerativeModel
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from textblob import TextBlob
from collections import Counter

# Page configuration
st.set_page_config(
    page_title="🎮 Nintendo AI Market Trend Advisor",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        backgroun
