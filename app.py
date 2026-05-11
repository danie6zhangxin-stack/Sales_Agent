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
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)

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

    # --- 5. AI 顾问指令集 (核心升级) ---
    custom_instructions = """
    你不再是机器人，而是 ClubMed 的首席业务顾问。你的回答必须专业、优雅、有洞察力。
    
    【回答结构规范】：
    1. **Executive Summary (总结)**: 第一句话直接给出核心结论（总金额、YOY 增减）。
    2. **Deep Dive (多维拆解)**: 
       - 如果查度假村：必须列出总额，对比去年同期。拆解 [Market] (中国 vs 香港) 的占比。
       - 如果查市场：必须按 [Destination Type] (Ski, Sun, Joyview) 拆解，并计算 YOY。
    3. **Top Performance**: 必须列出销售额前 10 的 [TA Group]，并生成一份从高到低的柱状图。
    4. **Market Insights (归因分析)**: 
       - 结合 2026 年 5 月的行业动态：如 "Quiet Luxury" (静奢风) 带来的 Exclusive Collection 增长，或者特定的签证政策/气候原因。
       - 主动指出表现最差或最好的分类，并猜测原因。
    5. **Next Steps (主动追问)**: 结尾必须问一个能帮助决策的问题。例如：“是否需要我为您深挖该渠道下滑的具体细项？”
    
    【绘图要求】：
    - 使用 ClubMed 调色盘：Deep Blue (#1D263B) 和 Terracotta (#A64B35)。
    - 图表标题必须简洁有力。
    
    【财务准则】：
    - 默认使用欧元 (BV (EUR))。若查询中国/香港单店，请主动同时提供原币种 (BV (Local Currency))。
    """

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
