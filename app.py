import streamlit as st
import pandas as pd
import numpy as np 
from pandasai import Agent
from langchain_openai import ChatOpenAI
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- 1. 高端商业极简风 UI 配置 (McKinsey x ClubMed) ---
st.set_page_config(page_title="ClubMed Executive Intelligence", layout="wide", page_icon="Ψ")

CSS_STYLE = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@300;400;500;600&display=swap');
    :root { --cm-blue: #1D263B; --cm-terracotta: #A64B35; --cm-sage: #A4B6B0; --cm-beige: #F8F9FA; }
    .main { background-color: var(--cm-beige); font-family: 'Inter', sans-serif; }
    h1, h2, h3, h4 { font-family: 'Playfair Display', serif !important; color: var(--cm-blue); }
    
    div[data-testid="stMetric"] { background-color: white; border-radius: 6px; padding: 15px 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.02); border-top: 3px solid var(--cm-terracotta); }
    div[data-testid="stMetricValue"] { color: var(--cm-blue); font-weight: 600; font-size: 28px; }
    .stDataFrame { border: 1px solid #EAECEF; border-radius: 6px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.01); background-color: white; }
    
    div[data-testid="stChatMessage"] { background-color: transparent !important; border: none !important; border-bottom: 1px solid #EAECEF !important; padding: 1.5rem 0.5rem !important; margin-bottom: 0 !important; }
    div[data-testid="stChatMessage"]:last-child { border-bottom: none !important; }
    div[data-testid="stChatMessageAvatarUser"] { background-color: var(--cm-blue) !important; color: white !important;}
    div[data-testid="stChatMessageAvatarAssistant"] { background-color: var(--cm-blue) !important; color: white !important;}
</style>
"""
st.markdown(CSS_STYLE, unsafe_allow_html=True)

# --- 2. DeepSeek AI 引擎初始化 ---
try:
    api_key = st.secrets["DEEPSEEK_API_KEY"]
except:
    api_key = "sk-xxxxxxxxxxxxxxxxxxxxxxxx" # ⚠️ 记得填写真实Key

llm = ChatOpenAI(api_key=api_key, base_url="https://api.deepseek.com", model="deepseek-chat", temperature=0.1)

# --- 3. 侧边栏与文件上传 ---
with st.sidebar:
    st.markdown("<h2 style='color:#A64B35; border-bottom: 1px solid #ddd; padding-bottom: 10px;'>ClubMed Ψ <br><span style='font-size:16px; font-family:Inter; color:#1D263B;'>Executive Dashboard</span></h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Data (CSV)", type=['csv'])
    st.divider()

# --- 4. 核心引擎与准确率保证 ---
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

    # 强制清理隐藏空格
    for c in ['Market', 'Resort', 'TA_Group', 'Dest_Type', 'Month']:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # 格式化财务数据
    for c in ['BV', 'HN']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
    df['ADR'] = (df['BV'] / df['HN']).replace([float('inf'), -float('inf')], 0).fillna(0)

    if 'Year' in df.columns:
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce').fillna(0).astype(int)

    # ==========================================
    # 🌟 模块 A & B：精准宏观 Dashboard 
    # ==========================================
    with st.sidebar:
        st.markdown("### ⚙️ Dashboard Filters")
        available_years = sorted([y for y in df['Year'].unique() if y > 2000], reverse=True)
        selected_year = st.selectbox("Select Consumption Year", available_years) if available_years else 2026

    st.markdown(f"### 📈 Executive Summary ({selected_year})")
    df_cy = df[df['Year'] == selected_year]
    df_py = df[df['Year'] == selected_year - 1]

    cy_bv, cy_hn = df_cy['BV'].sum(), df_cy['HN'].sum()
    py_bv, py_hn = df_py['BV'].sum(), df_py['HN'].sum()
    cy_adr = cy_bv / cy_hn if cy_hn > 0 else 0
    py_adr = py_bv / py_hn if py_hn > 0 else 0
    
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Total BV ({selected_year})", f"€ {cy_bv:,.0f}", f"{(cy_bv-py_bv)/py_bv*100:.1f}% vs {selected_year-1}" if py_bv>0 else None)
    c2.metric(f"Total HN ({selected_year})", f"{cy_hn:,.0f}", f"{(cy_hn-py_hn)/py_hn*100:.1f}% vs {selected_year-1}" if py_hn>0 else None)
    c3.metric(f"Avg ADR ({selected_year})", f"€ {cy_adr:,.2f}", f"{(cy_adr-py_adr)/py_adr*100:.1f}% vs {selected_year-1}" if py_adr>0 else None)

    st.markdown(f"#### 📊 China & HK Performance by Destination Type ({selected_year} vs {selected_year-1})")
    ch_hk_mask_cy = df_cy['Market'].str.contains('China|Hong Kong|HK', case=False, na=False)
    ch_hk_mask_py = df_py['Market'].str.contains('China|Hong Kong|HK', case=False, na=False)
    
    cy_target = df_cy[ch_hk_mask_cy].groupby(['Market', 'Dest_Type'])[['BV', 'HN']].sum().reset_index()
    py_target = df_py[ch_hk_mask_py].groupby(['Market', 'Dest_Type'])[['BV', 'HN']].sum().reset_index()
    
    dashboard_df = pd.merge(cy_target, py_target, on=['Market', 'Dest_Type'], how='outer', suffixes=(f'_{selected_year}', f'_{selected_year-1}')).fillna(0)
    dashboard_df[f'ADR_{selected_year}'] = np.where(dashboard_df[f'HN_{selected_year}'] > 0, dashboard_df[f'BV_{selected_year}'] / dashboard_df[f'HN_{selected_year}'], 0)
    dashboard_df[f'ADR_{selected_year-1}'] = np.where(dashboard_df[f'HN_{selected_year-1}'] > 0, dashboard_df[f'BV_{selected_year-1}'] / dashboard_df[f'HN_{selected_year-1}'], 0)
    dashboard_df['BV_YoY(%)'] = np.where(dashboard_df[f'BV_{selected_year-1}'] > 0, (dashboard_df[f'BV_{selected_year}'] - dashboard_df[f'BV_{selected_year-1}']) / dashboard_df[f'BV_{selected_year-1}'] * 100, 0)
    
    display_cols = ['Market', 'Dest_Type', f'BV_{selected_year}', f'BV_{selected_year-1}', 'BV_YoY(%)', f'HN_{selected_year}', f'ADR_{selected_year}']
    if not dashboard_df.empty:
        styled_dash = dashboard_df[display_cols].style.format({
            f'BV_{selected_year}': '€ {:,.0f}', f'BV_{selected_year-1}': '€ {:,.0f}',
            'BV_YoY(%)': '{:+.1f}%', f'HN_{selected_year}': '{:,.0f}', f'ADR_{selected_year}': '€ {:,.2f}'
        }).background_gradient(subset=['BV_YoY(%)'], cmap='RdYlGn', vmin=-20, vmax=20)
        st.dataframe(styled_dash, use_container_width=True, hide_index=True)
    else:
        st.info("No data found for China/Hong Kong in the selected year.")

    # ==========================================
    # 🌟 模块 C：具备“双模式”的智能 AI 顾问
    # ==========================================
    st.divider()
    st.markdown("### 🤖 Strategy Advisor")
    st.caption("Ask for a performance Deep Dive, or ask conversational questions like 'What is in the TA Group column?'")
    
    # 🌟 核心修复：引入“模式切换”，防止 AI 死板回答
    custom_instr = """
    You are a Senior Business Analyst for ClubMed. You have TWO operation modes depending on the user's prompt:

    === MODE 1: PERFORMANCE DEEP DIVE ===
    Trigger this mode when the user asks to analyze sales, BV, performance, or asks for a "deep dive".
    1. EXPLICIT YEAR FILTER (CRITICAL): If the user specifies a year (e.g., "2026"), you MUST explicitly filter the dataframe by that Year first (`df = df[df['Year'] == 2026]`). Do NOT mix 2025 and 2026 data unless specifically calculating YoY!
    2. EXACT MATCH FILTER: If requested a specific target (e.g. "NJ XXY"), use EXACT MATCH: `df = df[df['TA_Group'] == 'NJ XXY']`.
    3. MANDATORY FORMAT: You MUST return a MARKDOWN TEXT REPORT. DO NOT just return a raw Pandas DataFrame. 
       Your Markdown report MUST include:
       - Executive Summary (Total BV, HN, ADR for the specific period).
       - Breakdown by `Dest_Type` (Destination type Asia) using a Markdown table.
       - Breakdown by `Resort` using a Markdown table.
       - YoY Variance % (Compare current vs exact same period in Year-1).

    === MODE 2: Q&A AND DEBUGGING ===
    Trigger this mode when the user asks general questions, asks what filters were applied, or asks to explain logic (e.g., "Let me know what you filtered", "Why did you do that?").
    1. Answer directly in conversational plain text.
    2. Do NOT generate the long performance report.
    3. Note: The column 'group_TA_cml' has been renamed to `TA_Group` in the dataframe. If the user asks about it, explain the exact logic or list the unique values.

    === GENERAL RULES ===
    - CONVERSATIONAL MEMORY: Remember previous context. If previously asked about "NJ XXY", apply it to follow-ups.
    - CRITICAL DATATYPE RULE: ALWAYS cast your final numerical results to a String (e.g., `result = str(val)`) to prevent framework crashes.
    - ALWAYS include 'import numpy as np' and 'import pandas as pd'. NEVER generate matplotlib charts.
    """

    if "agent" not in st.session_state:
        st.session_state.agent = Agent(df, config={"llm": llm, "custom_instructions": custom_instr, "save_charts": False, "enable_cache": True})
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            if isinstance(m["content"], pd.DataFrame):
                st.dataframe(m["content"], use_container_width=True)
            else:
                st.markdown(m["content"])

    if prompt := st.chat_input("E.g., Deep dive into NJ XXY performance from Jan to May 2026."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Analyzing your request..."):
                try:
                    response = st.session_state.agent.chat(prompt)
                    
                    if isinstance(response, pd.DataFrame):
                        # 如果 AI 还是顽固地返回了表格，我们完美渲染它
                        st.markdown("**📊 Raw Data Table:**")
                        st.dataframe(response, use_container_width=True)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    else:
                        response_str = str(response)
                        st.markdown(response_str)
                        st.session_state.messages.append({"role": "assistant", "content": response_str})
                except Exception as e:
                    st.error(f"Analysis Error: {e}")

else:
    st.markdown("""
        <div style="height: 60vh; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center;">
            <h1 style="font-size: 3.5rem; margin-bottom: 1rem; color: #1D263B;">Data Meets Strategy.</h1>
            <p style="color: #6c757d; max-width: 600px; font-size: 1.1rem; line-height: 1.6;">
                Upload your secure sales dataset. Access McKinsey-standard Dashboards with Year Filters, Market/Dest_Type Pivots, and Conversational AI memory.
            </p>
        </div>
    """, unsafe_allow_html=True)
