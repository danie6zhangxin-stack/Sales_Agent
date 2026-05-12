import streamlit as st
import pandas as pd
import numpy as np 
from pandasai import Agent
from langchain_openai import ChatOpenAI
import matplotlib
matplotlib.use('Agg') # 强制离线渲染
import os

# --- 1. 高端商业极简风 UI 配置 (McKinsey Standard) ---
st.set_page_config(page_title="ClubMed Executive Intelligence", layout="wide", page_icon="Ψ")

CSS_STYLE = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@300;400;500;600&display=swap');
    :root { --cm-blue: #1D263B; --cm-terracotta: #A64B35; --cm-sage: #A4B6B0; --cm-beige: #F8F9FA; }
    
    .main { background-color: var(--cm-beige); font-family: 'Inter', sans-serif; }
    h1, h2, h3, h4 { font-family: 'Playfair Display', serif !important; color: var(--cm-blue); }
    
    /* KPI 指标卡片 */
    div[data-testid="stMetric"] { background-color: white; border-radius: 6px; padding: 15px 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.02); border-top: 3px solid var(--cm-terracotta); }
    div[data-testid="stMetricValue"] { color: var(--cm-blue); font-weight: 600; font-size: 28px; }
    
    /* 数据表格美化 */
    .stDataFrame { border: 1px solid #EAECEF; border-radius: 6px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.01); background-color: white; }
    
    /* 聊天气泡与头像的高定重塑 - 极致极简 */
    div[data-testid="stChatMessage"] { 
        background-color: transparent !important; 
        border: none !important; 
        border-bottom: 1px solid #EAECEF !important; 
        padding: 1.5rem 0.5rem !important; 
        margin-bottom: 0 !important; 
    }
    div[data-testid="stChatMessage"]:last-child { border-bottom: none !important; }
    
    /* 强制头像背景为深海蓝 */
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
    api_key = "sk-xxxxxxxxxxxxxxxxxxxxxxxx" # ⚠️ 填入你的真实Key

llm = ChatOpenAI(api_key=api_key, base_url="https://api.deepseek.com", model="deepseek-chat", temperature=0.1)

# --- 3. 侧边栏与文件上传 ---
with st.sidebar:
    st.markdown("<h2 style='color:#A64B35; border-bottom: 1px solid #ddd; padding-bottom: 10px;'>ClubMed Ψ <br><span style='font-size:16px; font-family:Inter; color:#1D263B;'>Executive Dashboard</span></h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload SalesData.csv", type=['csv'])
    st.divider()

# --- 4. 核心数据清洗引擎 ---
if uploaded_file:
    df = pd.read_csv(uploaded_file, low_memory=False)
    df.columns = [col.strip() for col in df.columns]

    # 规范化列名
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

    # 强制清理隐藏空格（彻底解决精确匹配失败的元凶）
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
    # 🌟 模块 A：原生 Python Dashboard (100% 准确率)
    # ==========================================
    with st.sidebar:
        st.markdown("### ⚙️ Dashboard Filters")
        available_years = sorted([y for y in df['Year'].unique() if y > 2000], reverse=True)
        selected_year = st.selectbox("Select Consumption Year", available_years) if available_years else 2026

    st.markdown(f"### 📈 Executive Summary ({selected_year} vs {selected_year-1})")
    df_cy = df[df['Year'] == selected_year]
    df_py = df[df['Year'] == selected_year - 1]

    cy_bv, cy_hn = df_cy['BV'].sum(), df_cy['HN'].sum()
    py_bv, py_hn = df_py['BV'].sum(), df_py['HN'].sum()
    cy_adr = cy_bv / cy_hn if cy_hn > 0 else 0
    py_adr = py_bv / py_hn if py_hn > 0 else 0
    
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Total BV", f"€ {cy_bv:,.0f}", f"{(cy_bv-py_bv)/py_bv*100:.1f}%" if py_bv>0 else None)
    c2.metric(f"Total HN", f"{cy_hn:,.0f}", f"{(cy_hn-py_hn)/py_hn*100:.1f}%" if py_hn>0 else None)
    c3.metric(f"Avg ADR", f"€ {cy_adr:,.2f}", f"{(cy_adr-py_adr)/py_adr*100:.1f}%" if py_adr>0 else None)

    # ==========================================
    # 🌟 模块 B：Destination Type 深度对比透视表
    # ==========================================
    st.markdown(f"#### 📊 China & HK Performance by Destination Type")
    
    ch_hk_mask_cy = df_cy['Market'].str.contains('China|Hong Kong|HK', case=False, na=False)
    ch_hk_mask_py = df_py['Market'].str.contains('China|Hong Kong|HK', case=False, na=False)
    
    cy_target = df_cy[ch_hk_mask_cy].groupby(['Market', 'Dest_Type'])[['BV', 'HN']].sum().reset_index()
    py_target = df_py[ch_hk_mask_py].groupby(['Market', 'Dest_Type'])[['BV', 'HN']].sum().reset_index()
    
    dashboard_df = pd.merge(cy_target, py_target, on=['Market', 'Dest_Type'], how='outer', suffixes=(f'_{selected_year}', f'_{selected_year-1}')).fillna(0)
    dashboard_df['BV_YoY(%)'] = np.where(dashboard_df[f'BV_{selected_year-1}'] > 0, (dashboard_df[f'BV_{selected_year}'] - dashboard_df[f'BV_{selected_year-1}']) / dashboard_df[f'BV_{selected_year-1}'] * 100, 0)
    
    display_cols = ['Market', 'Dest_Type', f'BV_{selected_year}', f'BV_{selected_year-1}', 'BV_YoY(%)']
    if not dashboard_df.empty:
        styled_dash = dashboard_df[display_cols].style.format({
            f'BV_{selected_year}': '€ {:,.0f}', f'BV_{selected_year-1}': '€ {:,.0f}', 'BV_YoY(%)': '{:+.1f}%'
        }).background_gradient(subset=['BV_YoY(%)'], cmap='RdYlGn', vmin=-15, vmax=15)
        st.dataframe(styled_dash, use_container_width=True, hide_index=True)

    # ==========================================
    # 🌟 模块 C：具备记忆力的智能 AI 战略顾问
    # ==========================================
    st.divider()
    st.markdown("### 🤖 Strategy Advisor (Conversational)")
    
    custom_instr = """
    You are a Senior McKinsey Consultant. Follow these STRICT rules:

    === OPERATION MODES ===
    Mode 1 (Deep Dive): For analysis/BV queries.
    Mode 2 (Q&A): For general questions about columns/logic.

    === CRITICAL RULES ===
    1. EXPLICIT YEAR FILTER: If user asks for 2026, YOU MUST filter `df = df[df['Year'] == 2026]` in your code.
    2. EXACT MATCH: If user asks for "NJ XXY", you MUST use EXACT matching `df = df[df['TA_Group'] == 'NJ XXY']`. DO NOT use str.contains()!
    3. MANDATORY DEST_TYPE: Every analysis MUST include a Markdown table breakdown by `Dest_Type`.
    4. NO PLOTTING (CRITICAL): NEVER generate charts, plots, or use matplotlib/seaborn. Output MUST be pure text and Markdown tables. 
    5. NO UNARYOP: Never use the '~' operator. Use '!= False' or '!= True' instead.
    6. DATATYPE: Cast all numerical results to String (e.g. `result = str(val)`).
    7. IMPORTS: Always include 'import numpy as np' and 'import pandas as pd'.
    """

    if "agent" not in st.session_state:
        # save_charts 必须为 False
        st.session_state.agent = Agent(df, config={"llm": llm, "custom_instructions": custom_instr, "save_charts": False, "enable_cache": True})
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 智能渲染历史对话
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            if isinstance(m["content"], pd.DataFrame): 
                st.dataframe(m["content"], use_container_width=True)
            elif isinstance(m["content"], str) and m["content"].endswith(".png"):
                st.image(m["content"])
            else: 
                st.markdown(m["content"])

    if prompt := st.chat_input("E.g., Deep dive into NJ XXY performance from Jan to May 2026."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.write(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Applying exact filters and executing deep dive..."):
                try:
                    response = st.session_state.agent.chat(prompt)
                    
                    # 🌟 智能排版与路径拦截器
                    if isinstance(response, pd.DataFrame):
                        st.markdown("**📊 Data Table Breakdown:**")
                        st.dataframe(response, use_container_width=True)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    elif isinstance(response, str) and response.endswith(".png"):
                        # 如果 AI 不听话非要画图，我们拦截路径并展示图片
                        if os.path.exists(response):
                            st.image(response)
                        else:
                            st.warning(f"AI generated a chart path, but file not found: {response}")
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
            <h1 style="font-size: 3.5rem; margin-bottom: 1rem; color: #1D263B;">Ψ Executive Intelligence</h1>
            <p style="color: #6c757d; max-width: 600px; font-size: 1.1rem; line-height: 1.6;">
                Ready for the boardroom. Upload your data to activate 100% accurate YoY Dashboards, Market/Dest_Type Pivots, and a Strategic AI with complete memory.
            </p>
        </div>
    """, unsafe_allow_html=True)
