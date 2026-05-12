import streamlit as st
import pandas as pd
import numpy as np 
from pandasai import Agent
from langchain_openai import ChatOpenAI
import matplotlib
matplotlib.use('Agg') # 强制离线渲染
import os

# --- 1. 高端商业视觉配置 (McKinsey x ClubMed Theme) ---
st.set_page_config(page_title="ClubMed Executive Intelligence", layout="wide", page_icon="Ψ")

CSS_STYLE = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@300;400;500;600&display=swap');
    :root { 
        --cm-blue: #1D263B;      /* 深海蓝 */
        --cm-terracotta: #A64B35; /* 陶土红 */
        --cm-beige: #F8F9FA;     
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
    .stDataFrame { border: 1px solid #EAECEF; border-radius: 6px; overflow: hidden; background-color: white; }
    
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
    api_key = "sk-xxxxxxxxxxxxxxxxxxxxxxxx" # ⚠️ 生产环境请确保配置了 Secrets

llm = ChatOpenAI(api_key=api_key, base_url="https://api.deepseek.com", model="deepseek-chat", temperature=0.1)

# --- 3. 核心数据清洗 (硬核去重与格式化) ---
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
    
    # 强制清理隐藏空格（解决 NJ XXY 匹配失败的元凶）
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
        
        markets = sorted([str(m) for m in df['Market'].unique() if str(m) != 'nan'])
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
        dash_df = df_cy[df_cy['Market'].isin(sel_markets)].groupby(['Market', 'Dest_Type'])[['BV', 'HN']].sum().reset_index()
        if not dash_df.empty:
            st.dataframe(dash_df.style.format({'BV': '€ {:,.0f}', 'HN': '{:,.0f}'}), use_container_width=True, hide_index=True)

    # 模块 C：智能 AI 决策顾问 (带记忆注入)
    st.divider()
    st.markdown("### 🤖 Strategy Advisor (Deep Dive)")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 提取历史对话作为记忆
    history_context = ""
    for msg in st.session_state.messages[-4:]:
        if isinstance(msg["content"], str):
            history_context += f"{msg['role']}: {msg['content'][:200]}\n"

    # 🌟 核心指令集：彻底解决数据错误与排版乱码
    custom_instr = f"""
    You are a McKinsey Consultant. Rules:
    1. EXACT MATCH (CRITICAL): If user asks for 'NJ XXY', you MUST use `df[df['TA_Group'] == 'NJ XXY']`. NEVER use fuzzy search.
    2. YEAR FILTER: Always apply `df[df['Year'] == 2026]` (or the requested year) first.
    3. DEST_TYPE DRILL-DOWN: Every analysis MUST include a breakdown by `Dest_Type` (Destination type Asia).
    4. OUTPUT FORMAT: ALWAYS return a clean pd.DataFrame. If you calculate a single number, turn it into a 1-row DataFrame. Round to 2 decimals.
    5. NO PLOTS: Do not use matplotlib.
    6. NO UNARYOP: Do not use '~'. Use '!= True' or '!= False'.
    7. RECENT CONTEXT: {history_context}
    """

    # 🌟 关键：原地创建 Agent 避开缓存 Bug
    agent = Agent(df, config={"llm": llm, "custom_instructions": custom_instr, "save_charts": False, "enable_cache": False})

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            if isinstance(m["content"], pd.DataFrame): st.dataframe(m["content"], use_container_width=True, hide_index=True)
            else: st.markdown(m["content"])

    if prompt := st.chat_input("E.g., Analyze NJ XXY monthly BV and Dest_Type breakdown for Jan-May 2026."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.write(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Executing precise deep dive..."):
                try:
                    response = agent.chat(prompt)
                    
                    if isinstance(response, pd.DataFrame) or type(response).__name__ == 'SmartDataframe':
                        if type(response).__name__ == 'SmartDataframe': response = response.dataframe
                        st.markdown("**📊 Deep Dive Results:**")
                        st.dataframe(response, use_container_width=True, hide_index=True)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    else:
                        res_str = str(response)
                        st.markdown(res_str)
                        st.session_state.messages.append({"role": "assistant", "content": res_str})
                except Exception as e:
                    st.error(f"Analysis Error: {e}")
else:
    st.markdown("<h1 style='text-align:center; padding-top:150px; opacity:0.2;'>Ψ Executive Hub</h1>", unsafe_allow_html=True)
