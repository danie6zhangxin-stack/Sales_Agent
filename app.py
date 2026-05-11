import streamlit as st
import pandas as pd
from pandasai import Agent
from langchain_openai import ChatOpenAI

# 1. Page Configuration
st.set_page_config(page_title="ClubMed AI Assistant", layout="wide", page_icon="Ψ")

# 2. Strict CSS Encapsulation (ClubMed Branding)
CSS_STYLE = """
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
"""
st.markdown(CSS_STYLE, unsafe_allow_html=True)

# 3. AI Engine Initialization (DeepSeek)
try:
    api_key = st.secrets["DEEPSEEK_API_KEY"]
except:
    api_key = "sk-xxxxxxxxxxxxxxxxxxxxxxxx" # Fallback if secrets fail

llm = ChatOpenAI(
    api_key=api_key, 
    base_url="https://api.deepseek.com", 
    model="deepseek-chat",
    temperature=0.1
)

# 4. Sidebar UI
with st.sidebar:
    st.markdown('<div class="cm-logo">ClubMed Ψ <br><span style="font-size:0.8rem; font-weight:normal; color:#9ca3af;">Management Intelligence</span></div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Sales Data (CSV)", type=['csv'])
    st.divider()
    st.markdown("### 💡 Recommended Prompts")
    st.caption("• Analyze the sales performance of Changbaishan in June 2026.")
    st.caption("• Why did the Beach category drop in Mainland China this May?")

# 5. Core Data Processing & AI Agent
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

    custom_instructions = """
    You are the Chief Strategy Advisor for ClubMed. You must respond entirely in ENGLISH.
    
    [Mandatory Response Structure]:
    1. **Executive Summary**: State Total BV (EUR and Local Currency).
    2. **Multi-dimensional Breakdown**: Breakdown by [Market]. Compare current data with [Consumption Year] minus 1 (YoY Variance %).
    3. **Top Performance**: List top 10 [TA Group] by sales. Output Python code to draw a horizontal bar chart (Color: #A64B35).
    4. **Strategic Attribution**: Briefly attribute sales changes to recent industry trends (e.g., 'Quiet Luxury', Visa changes).
    5. **Smart Follow-up**: End with a proactive question.
    """

    agent = Agent(df, config={
        "llm": llm,
        "custom_instructions": custom_instructions,
        "save_charts": False,
        "enable_cache": False,
        "enforce_privacy": False
    })

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
            with st.spinner("Analyzing data and generating insights..."):
                try:
                    response = agent.chat(prompt)
                    if isinstance(response, str):
                        st.markdown(response)
                    else:
                        st.write(response)
                    st.session_state.messages.append({"role": "assistant", "content": str(response)})
                except Exception as e:
                    st.error(f"Analysis Error: {e}")
else:
    st.markdown("""
        <div style="height: 60vh; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center;">
            <h1 style="font-size: 3.5rem; margin-bottom: 1.5rem;">Ψ Explore L'Esprit Libre</h1>
            <p style="color: #666; max-width: 600px; font-size: 1.2rem;">
                Welcome to the ClubMed Management Intelligence. Upload your sales data.
            </p>
        </div>
    """, unsafe_allow_html=True)
