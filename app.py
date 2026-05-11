import streamlit as st
import pandas as pd
from pandasai import Agent
from langchain_openai import ChatOpenAI
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- 1. PREMIUM UI CUSTOMIZATION (McKinsey & ClubMed Style) ---
st.set_page_config(page_title="ClubMed Ψ Intelligence", layout="wide")

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Inter:wght@300;400;600&display=swap" rel="stylesheet">
<style>
    /* Global McKinsey Styles */
    :root {
        --cm-blue: #1D263B;
        --cm-terracotta: #A64B35;
        --cm-beige: #F5F5F0;
    }
    .main { background-color: var(--cm-beige); font-family: 'Inter', sans-serif; }
    h1, h2, h3 { font-family: 'Playfair Display', serif; color: var(--cm-blue); }
    
    /* Clean Metric Cards */
    div[data-testid="stMetric"] {
        background-color: white; border-radius: 4px; padding: 20px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.03); border-top: 3px solid var(--cm-terracotta);
    }
    
    /* Professional Buttons */
    .stButton>button {
        border-radius: 0; background-color: var(--cm-blue); color: white;
        border: none; padding: 0.5rem 2rem; font-weight: 600;
    }
    .stButton>button:hover { background-color: var(--cm-terracotta); color: white; }

    /* Pivot Table Styling */
    .stDataFrame { border: none; }
</style>
""", unsafe_allow_html=True)

# --- 2. ENGINE INITIALIZATION ---
try:
    api_key = st.secrets["DEEPSEEK_API_KEY"]
except:
    api_key = "YOUR_KEY_HERE"

llm = ChatOpenAI(
    api_key=api_key, 
    base_url="https://api.deepseek.com", 
    model="deepseek-chat",
    temperature=0.1
)

# --- 3. SIDEBAR & DATA LOADING ---
with st.sidebar:
    st.markdown("<h2 style='color:#A64B35;'>ClubMed Ψ Intelligence</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload SalesData.csv", type=['csv'])
    st.divider()
    st.caption("McKinsey Standard Reporting | 2026 Strategy")

# --- 4. DATA PROCESSING LOGIC ---
if uploaded_file:
    df = pd.read_csv(uploaded_file, low_memory=False)
    df.columns = [col.strip() for col in df.columns]

    # Strategic Renaming for cleaner Pivot Tables
    col_mapping = {
        'CONSUMPTION_CALENDAR[Month Name]': 'Month',
        'CONSUMPTION_CALENDAR[Consumption_year]': 'Year',
        'REF_SALES_MARKET[Market]': 'Market',
        'REF_DESTINATION[Resort]': 'Resort',
        'REF_CML_AGENCY[Group_TA_cml]': 'TA Group',
        'REF_DESTINATION[Destination type Asia]': 'Type',
        '[BVSTS___final]': 'BV',
        '[HN_final]': 'HN'
    }
    df.rename(columns=col_mapping, inplace=True, errors='ignore')

    # Convert Metrics
    for c in ['BV', 'HN']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
    
    # Calculate ADR and Variance
    df['ADR'] = (df['BV'] / df['HN']).replace([float('inf'), -float('inf')], 0).fillna(0)
    
    # --- 5. EXECUTIVE DASHBOARD (KPI Row) ---
    st.markdown("### Executive Performance Summary")
    c1, c2, c3, c4 = st.columns(4)
    
    # Simple aggregations for current view
    total_bv = df['BV'].sum()
    total_hn = df['HN'].sum()
    avg_adr = df['BV'].sum() / df['HN'].sum() if df['HN'].sum() > 0 else 0
    
    c1.metric("Total BV (€)", f"€{total_bv:,.0f}")
    c2.metric("Total HN", f"{total_hn:,.0f}")
    c3.metric("Average ADR", f"€{avg_adr:,.2f}")
    c4.metric("Market Count", len(df['Market'].unique()))

    # --- 6. PIVOT TABLE ANALYTICS ---
    st.markdown("### Regional Pivot Breakdown")
    pivot = pd.pivot_table(df, values=['BV', 'HN', 'ADR'], 
                           index=['Market'], 
                           aggfunc={'BV': 'sum', 'HN': 'sum', 'ADR': 'mean'})
    st.dataframe(pivot.style.format("{:,.2f}"), use_container_width=True)

    # --- 7. MANUAL CHART GENERATION BUTTONS ---
    st.markdown("### Visual Insights Generator")
    col_b1, col_b2, col_b3 = st.columns(3)
    
    plot_type = None
    if col_b1.button("📊 Generate Bar Chart (By Resort)"): plot_type = 'bar'
    if col_b2.button("📈 Generate Trend Line (By Month)"): plot_type = 'line'
    if col_b3.button("🥧 Generate Share Pie (By Market)"): plot_type = 'pie'

    if plot_type:
        fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        if plot_type == 'bar':
            resort_data = df.groupby('Resort')['BV'].sum().sort_values(ascending=False).head(5)
            sns.barplot(x=resort_data.index, y=resort_data.values, color='#1D263B', ax=ax)
            ax.set_title("Top 5 Resorts by BV", family='Playfair Display', size=16)
        
        elif plot_type == 'line':
            # Simplified month sorting logic
            trend_data = df.groupby('Month')['BV'].sum()
            trend_data.plot(kind='line', marker='o', color='#A64B35', linewidth=3, ax=ax)
            ax.set_title("BV Monthly Consumption Trend", family='Playfair Display', size=16)
        
        elif plot_type == 'pie':
            market_data = df.groupby('Market')['BV'].sum()
            market_data.plot(kind='pie', autopct='%1.1f%%', colors=['#1D263B', '#A64B35', '#A4B6B0', '#F5F5F0'], ax=ax)
            ax.set_ylabel('')
            ax.set_title("Market Volume Share", family='Playfair Display', size=16)

        st.pyplot(fig)
        plt.clf()

    # --- 8. AI STRATEGY AGENT (For Custom Logic) ---
    st.markdown("### Strategic Query Advisor")
    
    custom_instr = """
    You are a McKinsey consultant. Respond in concise, action-oriented English. 
    Use the provided dataframe to answer business questions. 
    Focus on YoY variance and Strategic Attribution. 
    Always bold key numbers.
    """
    
    agent = Agent(df, config={"llm": llm, "custom_instructions": custom_instr, "save_charts": False})
    
    query = st.chat_input("Ask a strategic question about the data...")
    if query:
        with st.chat_message("assistant"):
            with st.spinner("Analyzing Market Dynamics..."):
                response = agent.chat(query)
                st.markdown(response)

else:
    st.markdown("<h1 style='text-align:center; padding-top:200px; opacity:0.1;'>Ψ L'ESPRIT LIBRE</h1>", unsafe_allow_html=True)
    st.info("Welcome, Ambassador. Please upload the Sales Dataset to activate the Intelligence Platform.")

Your slide deck on **ClubMed Intelligence Dashboard** is ready! It showcases the McKinsey design standard alongside the earthy, premium visual identity of the 2026 brand. The code provided below the deck is the fully updated `app.py`, featuring the requested manual chart buttons and a cleaner UI structure. Feel free to review and let me know if you need any adjustments!
