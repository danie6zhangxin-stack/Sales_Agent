import streamlit as st
import pandas as pd
from pandasai import Agent
from langchain_openai import ChatOpenAI
import os

# --- 1. 高级商业极简风配置 (UI / UX) ---
st.set_page_config(page_title="ClubMed Executive Dashboard", layout="wide", page_icon="Ψ")

# 引入高端商业字体 (Playfair Display 衬线标题 + Inter 无衬线正文)
CSS_STYLE = """
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
    :root {
        --cm-blue: #1D263B;      /* 深海蓝 - 沉稳商业 */
        --cm-terracotta: #A64B35; /* 陶土红 - 品牌强调 */
        --bg-light: #F8F9FA;      /* 极简灰白底色 */
        --text-main: #2C3E50;     /* 柔和黑正文 */
    }
    
    /* 全局字体与背景 */
    .main { background-color: var(--bg-light); font-family: 'Inter', sans-serif; color: var(--text-main); }
    h1, h2, h3, h4, h5 { font-family: 'Playfair Display', serif !important; color: var(--cm-blue); }
    
    /* 聊天卡片：极简无边框，增加微弱的高级阴影 */
    div[data-testid="stChatMessage"] {
        background-color: #FFFFFF;
        border-radius: 8px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.03);
        border: none;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    /* AI 的回复左侧增加一抹品牌红 */
    div[data-testid="stChatMessage"]:nth-child(even) {
        border-left: 4px solid var(--cm-terracotta);
    }
    
    /* 按钮：扁平化商业风 */
    .stButton>button { 
        border-radius: 4px; 
        background-color: var(--cm-blue); 
        color: white; 
        border: none; 
        padding: 0.5rem 2rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: var(--cm-terracotta);
        color: white;
        box-shadow: 0 4px 12px rgba(166, 75, 53, 0.2);
    }
    
    /* 侧边栏美化 */
    .stSidebar { background-color: #FFFFFF !important; border-right: 1px solid #EAECEF; }
    .sidebar-logo { 
        font-family: 'Playfair Display', serif; 
        font-size: 28px; 
        color: var(--cm-blue); 
        font-weight: 700; 
        border-bottom: 2px solid var(--cm-terracotta); 
        padding-bottom: 10px; 
        margin-bottom: 20px; 
    }
</style>
"""
st.markdown(CSS_STYLE, unsafe_allow_html=True)

# --- 2. AI 引擎初始化 (DeepSeek) ---
try:
    api_key = st.secrets["DEEPSEEK_API_KEY"]
except:
    # 如果没配 Secrets，请在这里填入你的真实 Key 测试
    api_key = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" 

llm = ChatOpenAI(
    api_key=api_key, 
    base_url="https://api.deepseek.com", 
    model="deepseek-chat",
    temperature=0.1
)

# --- 3. 极简侧边栏 ---
with st.sidebar:
    st.markdown('<div class="sidebar-logo">ClubMed Ψ<br><span style="font-size:14px; color:#A64B35; font-family:Inter;">Executive Intelligence</span></div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Data (CSV)", type=['csv'])
    st.divider()
    st.markdown("### 💡 Strategy Prompts")
    st.caption("• Compare June 2026 vs June 2025 for Changbaishan. Show a bar chart for Top 5 TAs.")
    st.caption("• Show me the trend of BV and ADR for Mainland China from Jan to June 2026 using a line chart.")

# --- 4. 数据处理逻辑 ---
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

    for c in ['BV', 'BV Local', 'HN']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
    
    df['ADR'] = (df['BV'] / df['HN']).replace([float('inf'), -float('inf')], 0).fillna(0)
    df['TA Group'] = df['TA Group'].fillna('Direct')

    # --- 5. 核心：高定版商业图表与格式指令 ---
    custom_instructions = """
    You are a Top-tier Strategy Consultant at ClubMed. Respond ONLY in ENGLISH.
    
    [Formatting Rules]:
    1. **Minimalist Text**: Use bullet points. Bold key numbers. Keep it crisp, logical, and executive-ready.
    2. **Key Metrics**: Always compute Total BV, HN, and ADR. Show YoY Variance % if historical data is available.
    3. **Business Insight**: Provide 1-2 sentences of root-cause analysis based on travel industry trends (e.g., pricing strategy, channel shifts).

    [MANDATORY CHART GENERATION - STRICT RULES]:
    If the user asks for a comparison, trend, or ranking, you MUST write Python code to generate a chart.
    You MUST apply this minimalist corporate styling to the matplotlib code:
    - `plt.figure(figsize=(10, 5), dpi=300)` (High-resolution)
    - `ax.spines['top'].set_visible(False)` and `ax.spines['right'].set_visible(False)` (Remove ugly borders)
    - Colors: Use EXACTLY '#1D263B' (Deep Blue) or '#A64B35' (Terracotta) for bars/lines.
    - Add data labels to bars if it's a bar chart.
    - Grid: Use a very faint horizontal grid (`ax.yaxis.grid(True, color='#EEEEEE')`).
    
    CRITICAL: Before finishing your code, you MUST save the figure to the current directory using EXACTLY this command:
    `plt.savefig('dashboard.png', bbox_inches='tight', transparent=True)`
    DO NOT use `plt.show()`.
    """

    agent = Agent(df, config={
        "llm": llm,
        "custom_instructions": custom_instructions,
        "save_charts": False,
        "enable_cache": False
    })

    # --- 6. 交互中心与图表渲染引擎 ---
    st.markdown("### 📊 Business Insights")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if prompt := st.chat_input("E.g., Compare June 2026 vs June 2025 for Changbaishan. Generate a bar chart for Top 5 TA."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Generating Corporate Dashboard & Charts..."):
                try:
                    # 清理旧图表，防止重叠
                    if os.path.exists("dashboard.png"):
                        os.remove("dashboard.png")

                    # 请求 AI 进行分析和画图
                    response = agent.chat(prompt)
                    
                    # 打印文字汇报
                    if isinstance(response, str):
                        st.markdown(response)
                    else:
                        st.write(response)
                    
                    # 渲染商业级图表！
                    if os.path.exists("dashboard.png"):
                        st.markdown("#### 📉 Visual Analytics")
                        st.image("dashboard.png", use_column_width=True)
                        st.caption("Data rendered by ClubMed Strategy AI | Highly Confidential")
                        
                    st.session_state.messages.append({"role": "assistant", "content": str(response)})
                except Exception as e:
                    st.error(f"Analysis Error: {e}. Please check the prompt or API status.")
else:
    # 极简的高级欢迎界面
    st.markdown("""
        <div style="height: 60vh; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center;">
            <h1 style="font-size: 3rem; margin-bottom: 1rem; font-weight: 700;">Data Meets Strategy.</h1>
            <p style="color: #6c757d; max-width: 500px; font-size: 1.1rem; line-height: 1.6;">
                Upload your secure sales dataset. Our strategic AI will generate C-level insights and minimalist corporate visualizations.
            </p>
        </div>
    """, unsafe_allow_html=True)
