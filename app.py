import streamlit as st
import pandas as pd
from pandasai import Agent
from langchain_google_genai import ChatGoogleGenerativeAI
import matplotlib.pyplot as plt

# --- 1. 极简品牌视觉定制 (ClubMed 2026 配色方案) ---
st.set_page_config(page_title="ClubMed AI Assistant Ψ", layout="wide", page_icon="Ψ")

st.markdown("""
    <style>
    /* 核心配色：Terracotta, Deep Blue, Sage Green, Beige */
    :root {
        --cm-blue: #1D263B;
        --cm-terracotta: #A64B35;
        --cm-sage: #A4B6B0;
        --cm-beige: #F5F5F0;
    }
    
    .main { background-color: var(--cm-beige); color: var(--cm-blue); }
    .stButton>button { border-radius: 50px; background-color: var(--cm-terracotta); color: white; border: none; padding: 0.5rem 2rem; }
    .stTextInput>div>div>input { border-radius: 15px; border: 1px solid var(--cm-sage); }
    
    /* 极简卡片样式 */
    div[data-testid="stExpander"] { background-color: white; border: none; border-radius: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); }
    
    /* 标题美化 */
    h1, h2, h3 { font-family: 'Playfair Display', serif; color: var(--cm-blue); }
    .cm-logo { font-size: 2rem; font-weight: bold; color: var(--cm-terracotta); margin-bottom: 2rem; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 安全初始化 AI 引擎 ---
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except:
    api_key = "YOUR_API_KEY"

# 使用最新的 Gemini 2.0/2.5 引擎
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)

# --- 3. 侧边栏设计 (极简风) ---
with st.sidebar:
    st.markdown('<div class="cm-logo">ClubMed Ψ <br><span style="font-size:0.8rem; font-weight:normal; color:#666;">L\'Esprit Libre Intelligence</span></div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload SalesData.csv", type=['csv'])
    st.divider()
    st.markdown("### 💡 建议查询场景")
    st.caption("• 长白山本季度表现如何？对比去年情况。")
    st.caption("• 分析中国市场，按度假村类型拆解并分析原因。")
    st.caption("• 哪个渠道导致了本月 Beach 类产品的下滑？")

# --- 4. 数据预处理逻辑 (财务级清洗) ---
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = [col.strip() for col in df.columns]

    # 字段映射
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

    # 格式化数字
    for c in ['BV (EUR)', 'BV (Local Currency)', 'HN']:
        if c in df.columns:
            df[c] = df[c].astype(str).str.replace(',', '').astype(float)
    
    df['TA Group'] = df['TA Group'].fillna('Direct Sales')

    # 5. 极简精准指令集 (防超载模式)
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

        # 配置 Agent：关闭所有会消耗额外算力的功能
        agent = Agent(df, config={
            "llm": llm,
            "custom_instructions": custom_instructions,
            "save_charts": False,
            "enforce_privacy": False,
            "enable_cache": False
        })

    # --- 6. 交互界面 ---
    st.markdown("### 📊 业务洞察看板")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if prompt := st.chat_input("询问 AI 顾问，例如：'分析 2026 年 6 月长白山的表现'"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("AI 顾问正在生成深度洞察..."):
                try:
                    # 运行 AI 逻辑
                    response = agent.chat(prompt)
                    
                    # 渲染结果
                    if isinstance(response, str):
                        st.markdown(response)
                    else:
                        st.write(response)
                    
                    st.session_state.messages.append({"role": "assistant", "content": str(response)})
                    
                    # 模拟“发送给老板”的按钮
                    st.button("✨ 生成管理层 Dashboard (PDF)")
                except Exception as e:
                    st.error(f"分析模块暂时不可用，请检查数据格式或 API 状态。错误详情: {e}")
else:
    # 极简欢迎页
    st.markdown("""
        <div style="height: 60vh; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center;">
            <h1 style="font-size: 3rem; margin-bottom: 1rem;">Experience L'Esprit Libre</h1>
            <p style="color: #666; max-width: 600px;">欢迎使用 ClubMed 销售智能助手。请上传最新的销售底稿，开始您的数据探索之旅。</p>
        </div>
    """, unsafe_allow_html=True)
