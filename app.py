import streamlit as st
import pandas as pd
import numpy as np  # <--- 修复点 1：全局强行引入 numpy，防止 AI 找不到
from pandasai import Agent
from langchain_openai import ChatOpenAI
import matplotlib
matplotlib.use('Agg') # 强制离线渲染，防止云端崩溃
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- 1. 高端商业极简风 UI 配置 (McKinsey x ClubMed) ---
st.set_page_config(page_title="ClubMed Executive Intelligence", layout="wide", page_icon="Ψ")

CSS_STYLE = """
<style>
    /* 引入高级字体 */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@300;400;500;600&display=swap');
    
    :root {
        --cm-blue: #1D263B;      
        --cm-terracotta: #A64B35; 
        --cm-sage: #A4B6B0;      
        --cm-beige: #F5F5F0;     
    }
    
    .main { background-color: var(--cm-beige); font-family: 'Inter', sans-serif; }
    h1, h2, h3 { font-family: 'Playfair Display', serif !important; color: var(--cm-blue); }
    
    /* 核心指标卡片美化 */
    div[data-testid="metric-container"] {
        background-color: white; border-radius: 4px; padding: 15px 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.02); border-top: 3px solid var(--cm-terracotta);
    }
    div[data-testid="stMetricValue"] { color: var(--cm-blue); font-weight: 600; font-size: 32px; }
    
    /* 透视表美化 */
    .stDataFrame { border: 1px solid #EAECEF; border-radius: 4px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.01); }
    
    /* 商业风按键 */
    .stButton>button { 
        border-radius: 2px; background-color: white; color: var(--cm-blue); 
        border: 1px solid var(--cm-blue); padding: 0.5rem 1rem; font-weight: 600; width: 100%; transition: all 0.3s ease;
    }
    .stButton>button:hover { background-color: var(--cm-blue); color: white; border-color: var(--cm-blue); }
    
    /* 聊天气泡与头像的高定视觉重塑 */
    div[data-testid="stChatMessage"] { 
        background-color: transparent !important; 
        border: none !important; 
        border-bottom: 1px solid #EAECEF !important; 
        padding: 1.5rem 0.5rem !important; 
        box-shadow: none !important; 
        margin-bottom: 0 !important;
    }
    div[data-testid="stChatMessage"]:last-child { border-bottom: none !important; }
    
    div[data-testid="stChatMessageAvatarUser"] { background-color: var(--cm-blue) !important; color: white !important;}
    div[data-testid="stChatMessageAvatarAssistant"] { background-color: var(--cm-blue) !important; color: white !important;}
    
    .stSidebar { background-color: white !important; border-right: 1px solid #EAECEF; }
</style>
"""
st.markdown(CSS_STYLE, unsafe_allow_html=True)

# --- 2. DeepSeek AI 引擎初始化 ---
try:
    api_key = st.secrets["DEEPSEEK_API_KEY"]
except:
    api_key = "sk-xxxxxxxxxxxxxxxxxxxxxxxx" # ⚠️ 请确保在后台配置了真实的 Key

llm = ChatOpenAI(
    api_key=api_key, 
    base_url="https://api.deepseek.com", 
    model="deepseek-chat",
    temperature=0.1
)

# --- 3. 极简侧边栏 ---
with st.sidebar:
    st.markdown("<h2 style='color:#A64B35; border-bottom: 1px solid #ddd; padding-bottom: 10px;'>ClubMed Ψ <br><span style='font-size:16px; font-family:Inter; color:#1D263B;'>Executive Dashboard</span></h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Data (CSV)", type=['csv'])
    st.divider()
    st.caption("Strategic Decision Platform | Featuring BV, HN, ADR analysis & Deep Dive reporting.")

# --- 4. 核心数据引擎 (财务级处理) ---
if uploaded_file:
    df = pd.read_csv(uploaded_file, low_memory=False)
    df.columns = [col.strip() for col in df.columns]

    col_mapping = {
        'CONSUMPTION_CALENDAR[Month Name]': 'Month',
        'CONSUMPTION_CALENDAR[Consumption_year]': 'Year',
        'REF_SALES_MARKET[Market]': 'Market',
        'REF_DESTINATION[Resort]': 'Resort',
        'REF_CML_AGENCY[Group_TA_cml]': 'TA_Group',
        'REF_DESTINATION[Destination type Asia]': 'Dest_Type',
        '[BVSTS___final]': 'BV',
        '[HN_final]': 'HN'
    }
    df.rename(columns=col_mapping, inplace=True, errors='ignore')

    for c in ['BV', 'HN']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
    
    df['ADR'] = (df['BV'] / df['HN']).replace([float('inf'), -float('inf')], 0).fillna(0)

    # --- 模块 A：全局指标看板 ---
    st.markdown("### 📈 Executive Summary")
    total_bv = df['BV'].sum()
    total_hn = df['HN'].sum()
    avg_adr = total_bv / total_hn if total_hn > 0 else 0
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Total BV (EUR)", f"€ {total_bv:,.0f}")
    c2.metric("Total HN (Nights)", f"{total_hn:,.0f}")
    c3.metric("Average ADR (EUR)", f"€ {avg_adr:,.2f}")

    # --- 模块 B：高级透视表 ---
    st.markdown("### 📊 Market Breakdown (Pivot)")
    pivot_df = pd.pivot_table(df, values=['BV', 'HN', 'ADR'], index=['Market'], aggfunc={'BV': 'sum', 'HN': 'sum', 'ADR': 'mean'})
    styled_pivot = pivot_df.style.format({'BV': '€ {:,.0f}', 'HN': '{:,.0f}', 'ADR': '€ {:,.2f}'})
    st.dataframe(styled_pivot, use_container_width=True)

    # --- 模块 C：一键生成图表 ---
    st.markdown("### 📉 Visual Analytics")
    b1, b2, b3 = st.columns(3)
    
    chart_to_draw = None
    if b1.button("📊 Top 5 Resorts (BV)"): chart_to_draw = 'bar'
    if b2.button("📈 Monthly Trend (BV)"): chart_to_draw = 'line'
    if b3.button("🥧 Market Share (BV)"): chart_to_draw = 'pie'

    if chart_to_draw:
        fig, ax = plt.subplots(figsize=(10, 4), dpi=200)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        if chart_to_draw == 'bar':
            resort_data = df.groupby('Resort')['BV'].sum().sort_values(ascending=False).head(5)
            sns.barplot(x=resort_data.index, y=resort_data.values, color='#A64B35', ax=ax)
            ax.set_title("Top 5 Resorts by BV", family='Playfair Display', color='#1D263B', size=16)
            ax.set_ylabel("BV (EUR)")
        
        elif chart_to_draw == 'line':
            trend_data = df.groupby('Month')['BV'].sum()
            trend_data.plot(kind='line', marker='o', color='#1D263B', linewidth=2.5, ax=ax)
            ax.set_title("BV Consumption Monthly Trend", family='Playfair Display', color='#1D263B', size=16)
            ax.set_ylabel("BV (EUR)")
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            
        elif chart_to_draw == 'pie':
            market_data = df.groupby('Market')['BV'].sum()
            market_data.plot(kind='pie', autopct='%1.1f%%', colors=['#1D263B', '#A64B35', '#A4B6B0', '#EAECEF'], ax=ax, textprops={'color':"w"})
            ax.set_ylabel('')
            ax.set_title("Volume Share by Market", family='Playfair Display', color='#1D263B', size=16)

        st.pyplot(fig)
        plt.clf()

    # --- 模块 D：AI 深度业务顾问 ---
    st.divider()
    st.markdown("### 🤖 Strategy Advisor (Deep Dive)")
    
    custom_instr = """
    You are a Senior McKinsey Consultant. For EVERY query about sales or BV performance, you MUST provide a response structured exactly as follows:

    1. **Executive Summary**: Total BV, HN, and ADR for the requested period/segment. 
    2. **Composition Breakdown**:
       - Markdown table showing the BV contribution By Resort.
       - Markdown table showing the breakdown By Dest_Type (Destination Type).
    3. **YoY Variance Analysis**:
       - Automatically fetch data for the SAME EXACT months in the previous year (Year - 1).
       - Show the Variance Amount and Variance %.

    CRITICAL DATA RULES:
    - ALWAYS include 'import numpy as np' and 'import pandas as pd' in your underlying python code to prevent NameError.
    - ALWAYS check exact string matches for Month (e.g., 'June' instead of 'JUN' or '06').
    - If user asks for 'Jan to May', sum the values for 'January', 'February', 'March', 'April', 'May'.
    - Use Markdown formatting exclusively for your final output. DO NOT attempt to generate matplotlib charts.
    """

    agent = Agent(df, config={"llm": llm, "custom_instructions": custom_instr, "save_charts": False, "enable_cache": False})

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if prompt := st.chat_input("E.g., Provide a deep dive analysis for NJ XXY. Jan to May 2026 vs last year."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Executing multi-dimensional deep dive..."):
                try:
                    response = agent.chat(prompt)
                    if not isinstance(response, str):
                        response = str(response)
                    
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Analysis Error: {e}")

else:
    st.markdown("""
        <div style="height: 60vh; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center;">
            <h1 style="font-size: 3rem; margin-bottom: 1rem; color: #1D263B;">Data Meets Strategy.</h1>
            <p style="color: #6c757d; max-width: 500px; font-size: 1.1rem; line-height: 1.6;">
                Upload your secure sales dataset. Access McKinsey-standard Dashboards, Pivot Analytics, and AI-driven YoY Deep Dives.
            </p>
        </div>
    """, unsafe_allow_html=True)
