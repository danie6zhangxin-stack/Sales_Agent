import streamlit as st
import pandas as pd
from pandasai import Agent
from langchain_google_genai import ChatGoogleGenerativeAI

# --- 1. 极简品牌视觉定制 ---
st.set_page_config(page_title="ClubMed AI Assistant Ψ", layout="wide", page_icon="Ψ")

st.markdown("""
    <style>
    :root {
        --cm-blue: #1D263B;
        --cm-terracotta: #A64B35;
        --cm-sage: #A4B6B0;
        --cm-beige: #F5F5F0;
    }
    .main { background-color: var(--cm-beige); color: var(--cm-blue); }
    .stButton>button { border-radius: 50px; background-color: var(--cm-terracotta); color: white; border: none; padding: 0.5rem 2rem; }
    .stTextInput>div>div>input { border-radius: 15px; border: 1px solid var(--cm-sage); }
    div[data-testid="stExpander"] { background-color: white; border: none; border-radius: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); }
    h1, h2, h3 { font-family: 'Playfair Display', serif; color: var(--cm-blue); }
    .cm-logo { font-size: 2rem; font-weight: bold; color: var(--cm-terracotta); margin-bottom: 2rem; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 安全初始化 AI 引擎 ---
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except:
    api_key = "YOUR_API_KEY"

# 使用 Google Gemini 2.0
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)

# --- 3. 侧边栏设计 ---
with st.sidebar:
    st.markdown('<div class="cm-logo">ClubMed Ψ <br><span style="font-size:0.8rem; font-weight:normal; color:#666;">L\'Esprit Libre Intelligence</span></div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload SalesData.csv", type=['csv'])

# --- 4. 数据预处理逻辑 ---
if uploaded_file:
    df = pd.read_csv(uploaded_file)
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

    # --- 5. 极简精准指令集 (防超载模式) ---
    custom_instructions = """
    你是 ClubMed 的数据计算引擎。请严格遵守以下规则，绝对不要生成任何图表，不要写长篇报告，不要做任何市场归因分析。
    
    【核心计算任务】：
    只要用户询问销售额，你必须直接、快速地计算并只输出以下内容：
    1. 当期销售数字 (Current BV)。
    2. 上一年同期的销售数字 (Previous Year BV，即 Consumption Year 减去 1)。
    3. 同比增减百分比 (Variance % = (当期 - 去年) / 去年 * 100%)。
    4. 如果用户特别要求了细分市场（例如中国和香港占比），简单列出数字和百分比即可。
    
    【财务准则】：
    - 默认计算 'BV (EUR)'。
    - 如果涉及中国区度假村（如长白山），必须同时计算 'BV (Local Currency)' (人民币)。
    - 数值保留两位小数，加上千分位逗号。回答尽量简短，像报表一样清晰。
    """

    agent = Agent(df, config={
        "llm": llm,
        "custom_instructions": custom_instructions,
        "save_charts": False,
        "enforce_privacy": False,
        "enable_cache": False
    })

    # --- 6. 交互界面 ---
    st.markdown("### 📊 业务数据速查看板")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if prompt := st.chat_input("询问如：2026年6月长白山的总销售额是多少？列出中国和香港占比。"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("AI 正在极速计算中..."):
                try:
                    response = agent.chat(prompt)
                    if isinstance(response, str):
                        st.markdown(response)
                    else:
                        st.write(response)
                    st.session_state.messages.append({"role": "assistant", "content": str(response)})
                except Exception as e:
                    st.error(f"计算出错: {e}")
else:
    st.markdown("<h2 style='text-align:center;margin-top:100px;'>请先上传数据文件</h2>", unsafe_allow_html=True)
