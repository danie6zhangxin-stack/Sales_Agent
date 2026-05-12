import streamlit as st
import pandas as pd
import numpy as np 
from pandasai import Agent
from langchain_openai import ChatOpenAI
import matplotlib
matplotlib.use('Agg') # 强制离线渲染
import os

# --- 1. 高端商业极简风 UI 配置 (McKinsey x ClubMed Theme) ---
st.set_page_config(page_title="ClubMed Executive Intelligence", layout="wide", page_icon="Ψ")

CSS_STYLE = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@300;400;500;600&display=swap');
    :root { 
        --cm-blue: #1D263B;      /* 深海蓝 */
        --cm-terracotta: #A64B35; /* 陶土红 */
        --cm-beige: #F8F9FA;     /* 极简底色 */
    }
    
    .main { background-color: var(--cm-beige); font-family: 'Inter', sans-serif; }
    h1, h2, h3, h4 { font-family: 'Playfair Display', serif !important; color: var(--cm-blue); }
    
    /* KPI 指标卡片 */
    div[data-testid="stMetric"] { 
        background-color: white; border-radius: 6px; padding: 15px 20px; 
        box-shadow: 0 2px 10px rgba(0,0,0,0.02); border-top: 3px solid var(--cm-terracotta); 
    }
    div[data-testid="stMetricValue"] { color: var(--cm-blue); font-weight: 600; font-size: 28px; }
    
    /* 数据表格美化 */
    .stDataFrame { border: 1px solid #EAECEF; border-radius: 6px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.01); background-color: white; }
    
    /* 聊天记录 - 极致极简流 */
    div[data-testid="stChatMessage"] { 
        background-color: transparent !important; 
        border: none !important; 
        border-bottom: 1px solid #EAECEF !important; 
        padding: 1.5rem 0.5rem !important; 
        margin-bottom: 0 !important; 
    }
    div[data-testid="stChatMessage"]:last-child { border-bottom: none !important; }
    
    /* 聊天头像 */
    div[data-testid="stChatMessageAvatarUser"], div[data-testid="stChatMessageAvatarAssistant"] { 
        background-color: var(--cm-blue) !important; color: white !important;
    }
    
    .stSidebar { background-color: white !important; border-right: 1px solid #EAECEF; }
</style>
"""
st.markdown(CSS_STYLE, unsafe_allow_html=True)

# --- 2. AI 引擎初始化 ---
try:
    api_key = st.secrets["DEEPSEEK_API_KEY"]
except:
    api_key = "sk-xxxxxxxxxxxxxxxxxxxxxxxx" # ⚠️ 填入你的真实Key

llm = ChatOpenAI(api_key=api_key, base_url="https://api.deepseek.com", model="deepseek-chat", temperature=0.1)

# --- 3. 核心数据清洗引擎 ---
@st.cache_data
def load_and_clean(file):
    data = pd.read_csv(file, low_memory=False)
    data.columns = [col.strip() for col in data.columns]
    mapping = {
        'CONSUMPTION_CALENDAR[Month Name]': 'Month',
        'CONSUMPTION_CALENDAR[Consumption_year]': 'Year',
        'REF_SALES_MARKET[Market]': 'Market',
        'REF_DESTINATION[Resort]': 'Resort',
        'REF_CML_AGENCY[Group_TA_cml]': 'TA_Group',
        'REF_DESTINATION[Destination type Asia]': 'Dest_Type',
        '[BVSTS___final]': 'BV',
        '[HN_final]': 'HN'
    }
    data.rename(columns=mapping, inplace=True, errors='ignore')
    
    # 强制清理隐藏空格
    for col in ['Market', 'Resort', 'TA_Group', 'Dest_Type', 'Month']:
        if col in data.columns:
            data[col] = data[col].astype(str).str.strip()
    
    # 财务数据转数字
    for col in ['BV', 'HN']:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
    
    data['ADR'] = (data['BV'] / data['HN']).replace([float('inf'), -float('inf')], 0).fillna(0)
    data['Year'] = pd.to_numeric(data['Year'], errors='coerce').fillna(0).astype(int)
    return data

# --- 4. 业务逻辑执行 ---
with st.sidebar:
    st.markdown("<h2 style='color:#A64B35;'>ClubMed Ψ Hub</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload SalesData.csv", type=['csv'])
    st.divider()

if uploaded_file:
    df = load_and_clean(uploaded_file)
    
    # 侧边栏动态筛选
    with st.sidebar:
        st.markdown("### ⚙️ Dashboard Filters")
        years = sorted([y for y in df['Year'].unique() if y > 2000], reverse=True)
        sel_year = st.selectbox("Select Year", years) if years else 2026
        
        markets = sorted([str(m) for m in df['Market'].unique() if str(m).strip() != '' and str(m).lower() != 'nan'])
        def_markets = [m for m in markets if any(k in m.lower() for k in ['china', 'hong kong', 'hk', 'cn'])]
        sel_markets = st.multiselect("Select Markets", markets, default=def_markets)

    # 模块 A：原生 100% 准确度看板
    st.markdown(f"### 📈 Executive Summary ({sel_year} vs {sel_year-1})")
    df_cy = df[df['Year'] == sel_year]
    df_py = df[df['Year'] == sel_year - 1]
    
    c1, c2, c3 = st.columns(3)
    cy_bv = df_cy['BV'].sum()
    py_bv = df_py['BV'].sum()
    c1.metric("Total BV", f"€ {cy_bv:,.0f}", f"{(cy_bv-py_bv)/py_bv*100:.1f}%" if py_bv>0 else None)
    c2.metric("Total HN", f"{df_cy['HN'].sum():,.0f}")
    c3.metric("Avg ADR", f"€ {df_cy['BV'].sum()/df_cy['HN'].sum() if df_cy['HN'].sum()>0 else 0:,.2f}")

    # 模块 B：市场与目的地拆解
    if sel_markets:
        st.markdown(f"#### 📊 Performance by Destination Type (Selected Markets)")
        df_cy_filtered = df_cy[df_cy['Market'].isin(sel_markets)]
        df_py_filtered = df_py[df_py['Market'].isin(sel_markets)]
        
        cy_target = df_cy_filtered.groupby(['Market', 'Dest_Type'])[['BV', 'HN']].sum().reset_index()
        py_target = df_py_filtered.groupby(['Market', 'Dest_Type'])[['BV', 'HN']].sum().reset_index()
        
        dash_df = pd.merge(cy_target, py_target, on=['Market', 'Dest_Type'], how='outer', suffixes=(f'_{sel_year}', f'_{sel_year-1}')).fillna(0)
        if not dash_df.empty:
            dash_df['BV_YoY(%)'] = np.where(dash_df[f'BV_{sel_year-1}'] > 0, 
                                           (dash_df[f'BV_{sel_year}'] - dash_df[f'BV_{sel_year-1}']) / dash_df[f'BV_{sel_year-1}'] * 100, 0)
            
            display_cols = ['Market', 'Dest_Type', f'BV_{sel_year}', f'BV_{sel_year-1}', 'BV_YoY(%)']
            styled_dash = dash_df[display_cols].style.format({
                f'BV_{sel_year}': '€ {:,.0f}', f'BV_{sel_year-1}': '€ {:,.0f}', 'BV_YoY(%)': '{:+.1f}%'
            }).background_gradient(subset=['BV_YoY(%)'], cmap='RdYlGn', vmin=-15, vmax=15)
            
            st.dataframe(styled_dash, use_container_width=True, hide_index=True)
        else:
            st.info("No destination data available for the selected Markets in this year.")

    # ==========================================
    # 🌟 模块 C：深度分析生成器 (强制输出原生 DataFrame)
    # ==========================================
    st.divider()
    st.markdown("### 🤖 Strategy Advisor (Deep Dive Table Generator)")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 提取历史对话作为记忆
    history_context = ""
    if len(st.session_state.messages) > 0:
        history_context = "\n\n=== RECENT CONVERSATION HISTORY ===\n"
        for msg in st.session_state.messages[-4:]:
            if isinstance(msg["content"], str) and not msg["content"].endswith(".png"):
                history_context += f"{msg['role'].upper()}: {msg['content'][:300]}...\n"

    # 🌟 终极防呆指令集：绝对匹配月份 + 强制返回数据表
    custom_instr = f"""
    You are a Senior Data Analyst for ClubMed. Follow these STRICT rules:

    === CORE FILTERING RULES ===
    1. EXPLICIT YEAR FILTER: Always filter by the requested year first (e.g., `df = df[df['Year'] == 2026]`).
    2. EXACT MATCH: If asked for a target like "NJ XXY", you MUST use EXACT matching `df = df[df['TA_Group'] == 'NJ XXY']`. DO NOT use str.contains()!
    3. MONTH FORMAT (CRITICAL): The `Month` column contains FULL capitalized names (e.g., 'January', 'February'). NEVER use abbreviations like 'Jan' or 'Feb' for filtering. If user says 'Jan to May', you MUST use `df['Month'].isin(['January', 'February', 'March', 'April', 'May'])`.

    === CRITICAL OUTPUT FORMAT (MUST OBEY) ===
    1. YOU MUST ALWAYS RETURN A PANDAS DATAFRAME (`pd.DataFrame`). 
    2. NEVER return raw text, strings, dictionaries, or raw Series. 
    3. If asked for a deep dive (e.g. Month and Dest_Type), group the data, calculate BV, HN, and ADR, and convert the result into a clean, flat DataFrame using `.reset_index()`.
    4. Rename columns to be business-friendly (e.g., 'Month', 'Destination Type', 'Business Volume').
    5. Round numerical values to 2 decimal places.
    6. NO PLOTTING: Never use matplotlib or seaborn.
    
    === CONTEXT ===
    Use this recent conversation context if the user asks a follow-up question:
    {history_context}
    """

    # 原地创建 Agent 避开缓存 Bug
    agent = Agent(df, config={"llm": llm, "custom_instructions": custom_instr, "save_charts": False, "enable_cache": False})

    # 渲染历史
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            if isinstance(m["content"], pd.DataFrame): 
                st.dataframe(m["content"], use_container_width=True, hide_index=True)
            elif isinstance(m["content"], str) and m["content"].endswith(".png"):
                st.image(m["content"])
            else: 
                st.markdown(m["content"])

    if prompt := st.chat_input("E.g., Show me NJ XXY's BV and HN from Jan to May 2026 broken down by Month and Dest_Type."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.write(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Generating precise multi-dimensional data table..."):
                try:
                    response = agent.chat(prompt)
                    
                    # 强行兜底渲染为高级交互表格
                    if isinstance(response, pd.DataFrame) or type(response).__name__ == 'SmartDataframe':
                        if type(response).__name__ == 'SmartDataframe': response = response.dataframe
                        st.markdown("**📊 Deep Dive Analysis Table:**")
                        st.dataframe(response, use_container_width=True, hide_index=True)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    else:
                        res_str = str(response)
                        st.markdown(res_str)
                        st.session_state.messages.append({"role": "assistant", "content": res_str})
                except Exception as e:
                    st.error(f"Analysis Error: {e}")
else:
    st.markdown("""
        <div style="height: 60vh; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center;">
            <h1 style="font-size: 3.5rem; margin-bottom: 1rem; color: #1D263B;">Ψ Executive Hub</h1>
            <p style="color: #6c757d; max-width: 600px; font-size: 1.1rem; line-height: 1.6;">
                Ready for the boardroom. Upload your data to activate 100% accurate Dashboards and an AI Data Analyst that delivers perfect tables.
            </p>
        </div>
    """, unsafe_allow_html=True)
