import streamlit as st
import pandas as pd
from pandasai import Agent
from langchain_openai import ChatOpenAI
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Page Configuration & Professional Theme ---
st.set_page_config(page_title="ClubMed Executive Dashboard", layout="wide", page_icon="Ψ")

CSS_STYLE = """
<style>
    :root {
        --cm-blue: #1D263B; --cm-terracotta: #A64B35; --cm-sage: #A4B6B0; --cm-beige: #F5F5F0;
    }
    .main { background-color: var(--cm-beige); font-family: 'Arial', sans-serif; }
    /* Dashboard Metric Card */
    .metric-container {
        background-color: white; padding: 20px; border-radius: 15px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05); border-top: 5px solid var(--cm-terracotta);
        text-align: center;
    }
    .stButton>button { border-radius: 50px; background-color: var(--cm-terracotta); color: white; }
    h1, h2, h3 { color: var(--cm-blue); }
    .cm-logo { font-size: 2rem; font-weight: bold; color: var(--cm-terracotta); margin-bottom: 1rem; }
</style>
"""
st.markdown(CSS_STYLE, unsafe_allow_html=True)

# --- 2. AI Engine (DeepSeek) ---
try:
    api_key = st.secrets["DEEPSEEK_API_KEY"]
except:
    api_key = "sk-你的真实DEEPSEEK密钥" # ⚠️ 也可以直接在这里填入测试

llm = ChatOpenAI(
    api_key=api_key, 
    base_url="https://api.deepseek.com", 
    model="deepseek-chat",
    temperature=0.1
)

# --- 3. Sidebar ---
with st.sidebar:
    st.markdown('<div class="cm-logo">ClubMed Ψ Dashboard</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Sales CSV", type=['csv'])
    st.info("💡 Pro Tip: Ask for 'Trend Analysis' to get line charts or 'Market Share' for pie charts.")

# --- 4. Data Processing (Including ADR Calculation) ---
if uploaded_file:
    df = pd.read_csv(uploaded_file, low_memory=False)
    df.columns = [col.strip() for col in df.columns]

    col_mapping = {
        'CONSUMPTION_CALENDAR[Month Name]': 'Month',
        'CONSUMPTION_CALENDAR[Consumption_year]': 'Year',
        'REF_SALES_MARKET[Market]': 'Market',
        'REF_DESTINATION[Resort]': 'Resort',
        'REF_CML_AGENCY[Group_TA_cml]': 'TA Group',
        'REF_DESTINATION[Destination type Asia]': 'Dest Type',
        '[BVSTS___final]': 'BV',
        '[BVSTS_loc_final]': 'BV Local',
        '[HN_final]': 'HN'
    }
    df.rename(columns=col_mapping, inplace=True)

    # 财务数据清理
    for c in ['BV', 'BV Local', 'HN']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
    
    # 【核心逻辑】：自动计算 ADR (BV / HN)
    df['ADR'] = (df['BV'] / df['HN']).replace([float('inf'), -float('inf')], 0).fillna(0)
    df['TA Group'] = df['TA Group'].fillna('Direct')

    # --- 5. Custom Instructions for Executive Output ---
    custom_instructions = """
    You are the Chief Financial Analyst at ClubMed. 
    When asked about performance, you MUST provide a structured 'Executive Dashboard' in English:

    1. **Key Metrics Row**: Provide Total BV, Total HN, and Average ADR. 
       Compare them with the Same Period Last Year (SPLY) and calculate Variance %.
    2. **Visual Analytics**:
       - For TA or Resort ranking: Generate a **Bar Chart** (Color: #A64B35).
       - For Trend analysis (by month): Generate a **Line Chart** showing monthly changes.
       - For Market share: Generate a **Pie Chart**.
    3. **Deep Insight**: Explain the correlation between BV and ADR. (e.g., 'The increase in BV was driven by higher ADR despite lower HN').
    4. **Formatting**: Use bold fonts for figures. Use tables for breakdowns between Mainland China and HK.
    
    IMPORTANT: Use 'matplotlib' for all charts. Always set the figure size to (10, 5).
    """

    agent = Agent(df, config={
        "llm": llm,
        "custom_instructions": custom_instructions,
        "save_charts": False,
        "enable_cache": False,
        "verbose": True # 方便调试看代码
    })

    # --- 6. Interface ---
    st.markdown("### 📈 Management Strategy Center")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if prompt := st.chat_input("E.g., Compare June 2026 vs June 2025 for Changbaishan. Show BV, HN, ADR and trends."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Generating Dashboard..."):
                try:
                    # 获取 AI 回复
                    response = agent.chat(prompt)
                    
                    # 检查是否有图表生成 (Matplotlib)
                    fig = plt.gcf()
                    
                    # 显示文字内容
                    st.markdown(response)
                    
                    # 如果有图表（且不是空的），显示图表
                    if fig.get_axes():
                        st.pyplot(fig)
                        plt.clf() # 清除缓存防止重叠
                    
                    st.session_state.messages.append({"role": "assistant", "content": str(response)})
                except Exception as e:
                    st.error(f"Error: {e}. Please ensure your DeepSeek API Key is valid and has enough credits.")
else:
    st.markdown("<h2 style='text-align:center;margin-top:100px;'>Please upload SalesData.csv to activate Dashboard</h2>", unsafe_allow_html=True)
