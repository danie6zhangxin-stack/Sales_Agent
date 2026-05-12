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

# --- 1. 高端商业视觉配置 (ClubMed Premium Theme) ---
st.set_page_config(page_title="ClubMed Executive Intelligence", layout="wide", page_icon="Ψ")

CSS_STYLE = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@300;400;500;600&display=swap');
    :root { 
        --cm-blue: #1D263B;      /* 深海蓝 */
        --cm-terracotta: #A64B35; /* 陶土红 */
        --cm-sage: #A4B6B0;      /* 鼠尾草绿 */
        --cm-beige: #F8F9FA;     /* 极简底色 */
    }
    
    .main { background-color: var(--cm-beige); font-family: 'Inter', sans-serif; }
    h1, h2, h3, h4 { font-family: 'Playfair Display', serif !important; color: var(--cm-blue); }
    
    /* KPI 仪表盘卡片 */
    div[data-testid="stMetric"] { 
        background-color: white; border-radius: 4px; padding: 15px 20px; 
        box-shadow: 0 4px 15px rgba(0,0,0,0.02); border-top: 4px solid var(--cm-terracotta); 
    }
    div[data-testid="stMetricValue"] { color: var(--cm-blue); font-weight: 600; font-size: 30px; }
    
    /* 透视表样式 */
    .stDataFrame { border: 1px solid #EAECEF; border-radius: 4px; overflow: hidden; background-color: white; }
    
    /* 聊天界面极简重塑 */
    div[data-testid="stChatMessage"] { 
        background-color: transparent !important; border: none !important; 
        border-bottom: 1px solid #EAECEF !important; padding: 1.5rem 0.5rem !important; 
    }
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
    api_key = "sk-xxxxxxxxxxxxxxxxxxxxxxxx" 

llm = ChatOpenAI(api_key=api_key, base_url="https://api.deepseek.com", model="deepseek-chat", temperature=0.1)

# --- 3. 数据清洗与格式化 ---
@st.cache_data
def load_and_clean_data(file):
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
    
    # 清洗文本和数字
    for col in ['Market', 'Resort', 'TA_Group', 'Dest_Type', 'Month']:
        if col in data.columns:
            data[col] = data[col].astype(str).str.strip()
    
    for col in ['BV', 'HN']:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
    
    data['ADR'] = (data['BV'] / data['HN']).replace([float('inf'), -float('inf')], 0).fillna(0)
    data['Year'] = pd.to_numeric(data['Year'], errors='coerce').fillna(0).astype(int)
    return data

# --- 4. 核心逻辑执行 ---
with st.sidebar:
    st.markdown("<h2 style='color:#A64B35;'>ClubMed Ψ Intelligence</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload SalesData.csv", type=['csv'])

if uploaded_file:
    df = load_and_clean_data(uploaded_file)
    
    # 侧边栏年份筛选
    with st.sidebar:
        st.markdown("### ⚙️ Global Filter")
        available_years = sorted([y for y in df['Year'].unique() if y > 2000], reverse=True)
        selected_year = st.selectbox("Current Reporting Year", available_years)

    # 顶部 Dashboard (McKinsey Style)
    st.markdown(f"### 📈 Executive Performance Dashboard ({selected_year})")
    df_cy = df[df['Year'] == selected_year]
    df_py = df[df['Year'] == selected_year - 1]
    
    c1, c2, c3 = st.columns(3)
    cy_bv = df_cy['BV'].sum()
    py_bv = df_py['BV'].sum()
    c1.metric("Total BV", f"€ {cy_bv:,.0f}", f"{(cy_bv-py_bv)/py_bv*100:.1f}% vs LY" if py_bv>0 else None)
    c2.metric("Total HN", f"{df_cy['HN'].sum():,.0f}")
    c3.metric("Avg ADR", f"€ {df_cy['BV'].sum()/df_cy['HN'].sum() if df_cy['HN'].sum()>0 else 0:,.2f}")

    # ==========================================
    # 🌟 模块 C：具备记忆力的深度分析顾问
    # ==========================================
    st.divider()
    st.markdown("### 🤖 Strategy AI Advisor")
    
    # 强制配色方案指令
    custom_instr = f"""
    You are a McKinsey Consultant for ClubMed. Follow these IRREVERSIBLE rules:

    1. **CLUBMED VISUALS (CRITICAL)**: Any chart you create MUST use color='#1D263B' (Deep Blue) or '#A64B35' (Terracotta). No default colors.
    2. **NJ XXY PRECISION**: If asked about "NJ XXY" in 2026, you MUST filter exactly by `TA_Group == 'NJ XXY'` and `Year == 2026`. Do NOT include other NJ agencies.
    3. **DEST_TYPE DRILL-DOWN**: Every analysis MUST include a breakdown by `Dest_Type` (Destination type Asia). Show BV, HN, and ADR for each category.
    4. **MONTHLY TRANSPARENCY**: When asked for a period (e.g., Jan-May), provide a monthly breakdown table to show how the total is achieved.
    5. **CONVERSATIONAL MEMORY**: Remember the TA Group and period from the previous question.
    6. **FORMATTING**: Use Markdown tables. Cast numbers to strings to prevent crashes.
    """

    if "agent" not in st.session_state:
        st.session_state.agent = Agent(df, config={
            "llm": llm, 
            "custom_instructions": custom_instr,
            "save_charts": True,
            "enable_cache": True
        })
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            if isinstance(m["content"], pd.DataFrame): st.dataframe(m["content"], use_container_width=True)
            else: st.markdown(m["content"])

    if prompt := st.chat_input("Analyze NJ XXY from Jan to May 2026. Show monthly BV and Dest Type breakdown."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.write(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Decoding ClubMed performance data..."):
                try:
                    response = st.session_state.agent.chat(prompt)
                    
                    if isinstance(response, pd.DataFrame):
                        st.dataframe(response, use_container_width=True)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    else:
                        response_str = str(response)
                        st.markdown(response_str)
                        # 检查是否有图表生成并渲染
                        chart_path = "exports/charts/temp_chart.png" # PandasAI 默认路径
                        if os.path.exists(chart_path):
                            st.image(chart_path)
                        st.session_state.messages.append({"role": "assistant", "content": response_str})
                except Exception as e:
                    st.error(f"Analysis Error: {e}")
else:
    st.markdown("<h1 style='text-align:center; padding-top:150px; opacity:0.1;'>Ψ L'ESPRIT LIBRE</h1>", unsafe_allow_html=True)
