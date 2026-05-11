import streamlit as st
import pandas as pd
from pandasai import Agent
from langchain_openai import ChatOpenAI
import matplotlib.pyplot as plt

# --- 1. ClubMed "L'Esprit Libre" Minimalist Branding ---
st.set_page_config(page_title="ClubMed AI Assistant Ψ", layout="wide", page_icon="Ψ")

st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;1,700&family=Urbanist:wght@300;400;500;700&display=swap" rel="stylesheet">
    <style>
    :root {
        --cm-blue: #1D263B;      
        --cm-terracotta: #A64B35; 
        --cm-sage: #A4B6B0;      
        --cm-beige: #F5F5F0;     
        --cm-white: #FFFFFF;
    }
    
    .main { background-color: var(--cm-beige); color: var(--cm-blue); font-family: 'Urbanist', sans-serif; }
    
    div[data-testid="stChatMessage"] {
        background-color: var(--cm-white);
        border-radius: 15px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.03);
        margin-bottom: 1.5rem;
        border-left: 5px solid var(--cm-sage);
    }
    
    .stSidebar { background-color: var(--cm-blue) !important; color: white !important; }
    .stSidebar [data-testid="stMarkdownContainer"] p { color: #d1d5db !important; }
    .stButton>button { 
        border-radius: 50px; 
        background-color: var(--cm-terracotta); 
        color: white; 
        border: none; 
        padding: 0.6rem 2.5rem;
        font-weight: 600;
    }
    
    h1, h2, h3 { font-family: 'Playfair Display', serif !important; color: var(--cm-blue); }
    .cm-logo { font-size: 2.2rem; font-weight: bold; color: var(--cm-terracotta); margin-bottom: 2rem; border-bottom: 1px solid #374151; padding-bottom: 1rem; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. Initialize DeepSeek Engine ---
try:
    api_key = st.secrets["DEEPSEEK_API_KEY"]
except:
    api_key = "sk-xxxxxxxxxxxxxxxxxxx" # ⚠️ REPLACE WITH YOUR DEEPSEEK API KEY

llm = ChatOpenAI(
    api_key=api_key, 
    base_url="https://api.deepseek.com", 
    model="deepseek-chat",
    temperature=0.2 
)

# --- 3. Minimalist Sidebar ---
with st.sidebar:
    st.markdown('<div class="cm-logo">ClubMed Ψ <br><span style="font-size:0.8rem; font-weight:normal; color:#9ca3af;">Management Intelligence</span></div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Sales Data (CSV)", type=['csv'])
    st.divider()
    st.markdown("### 💡 Recommended Prompts")
    st.caption("• Analyze the sales performance of Changbaishan in June 2026.")
    st.caption("• Why did the Beach category drop in Mainland China this May?")
    st.caption("• Generate a YTD business summary for Greater China.")

# --- 4. Data Pre-processing ---
if uploaded_file:
    df = pd.read_csv(uploaded_file, low_memory=False)
    df.columns = [col.strip() for col in df.columns]

    col_mapping = {
        'CONSUMPTION_CALENDAR[Month Name]': 'Consumption Month',
        'CONSUMPTION_CALENDAR[Consumption_year]': 'Consumption Year',
        'REF_SALES_MARKET[Market]': 'Market',
        'REF_DESTINATION[Resort]': 'Resort',
        'REF_CML_AGENCY[Group_TA_cml]': 'TA Group',
        'REF_DESTINATION[Destination type Asia]': 'Destination Type',
        '[BVSTS___final]': 'BV (EUR)',
        '[BVSTS_loc_final]': 'BV (Local Currency)',
        '[HN_final]': 'HN'
    }
    df.rename(columns=col_mapping, inplace=True)

    for c in ['BV (EUR)', 'BV (Local Currency)', 'HN']:
        if c in df.columns:
            df[c] = df[c].astype(str).str.replace(',', '').astype(float)
    df['TA Group'] = df['TA Group'].fillna('Direct Sales')

    # --- 5. L'Esprit Libre Logic (ALL ENGLISH) ---
    custom_instructions = """
    You are the Chief Strategy Advisor for ClubMed. You must respond entirely in ENGLISH. 
    Your tone should be highly professional, elegant, and insightful.
    
    [Mandatory Response Structure]:
    1. **Executive Summary**: Clearly state the Total BV (in EUR, and add Local Currency if related to China).
    2. **Multi-dimensional Breakdown**: 
       - Breakdown by [Market]: Mainland China vs. Hong Kong.
       - YoY (Year-over-Year) Variance: Compare current data with [Consumption Year] minus 1. Provide the Variance %.
    3. **Top Performance & Visualization**: 
       - List the top 10 [TA Group] by sales.
       - **MANDATORY CHART**: Write Python code to generate a horizontal bar chart for the Top 10 TA. Use color '#A64B35' (Terracotta) for the bars.
    4. **Category Insight (Only for Macro Queries)**:
       - Breakdown by [Destination Type] (Mountain, Beach, Countryside).
    5. **Strategic Attribution (Business Context)**:
       - Assume the current context is May 2026. Briefly attribute the sales performance to industry trends (e.g., 'Quiet Luxury', Visa policy changes, or extended ski seasons). Provide a strategic guess on why certain channels or categories increased/decreased.
    6. **Smart Follow-up**: End with a proactive strategic question for the user.
    
    [Formatting]: Use Markdown tables for data. Bold key figures. Do not write a massive essay, keep it concise and structured like a C-level dashboard.
    """

    agent = Agent(df, config={
        "llm": llm,
        "custom_instructions": custom_instructions,
        "save_charts": False,
        "enable_cache": False,
        "enforce_privacy": False
    })

    # --- 6. Interactive Dashboard ---
    st.markdown("### 📊 Executive Decision Center")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if prompt := st.chat_input("E.g., Analyze the sales performance of Changbaishan in June 2026."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing data and generating strategic insights..."):
                try:
                    response = agent.chat(prompt)
                    
                    if isinstance(response, str):
                        st.markdown(response)
                    else:
                        st.write(response)
                    
                    st.session_state.messages.append({"role": "assistant", "content": str(response)})
                    st.divider()
                    st.button("📄 Export Dashboard as PDF")
                except Exception as e:
                    st.error(f"Analysis failed. Please check the API connection. Error: {e}")
else:
    # Minimalist Welcome Screen
    st.markdown("""
        <div style="height: 60vh; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center;">
            <h1 style="font-size: 3.5rem; margin-bottom: 1.5rem;">Ψ Explore L'Esprit Libre</h1>
            <p style="color: #666; max-width: 600px; font-size: 1.2rem;">
                Welcome to the ClubMed Management Intelligence. Upload your sales data to access deep financial analytics and strategic insights.
            </p>
        </div>
    """, unsafe_allow_html=True)
