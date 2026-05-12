import streamlit as st
import pandas as pd
import numpy as np 
from pandasai import Agent
from langchain_openai import ChatOpenAI
# 🌟 修复点：更新为 LangChain 的最新核心组件路径
from langchain_core.messages import SystemMessage, HumanMessage
import plotly.graph_objects as go
import os

# --- 1. 高端商业视觉配置 (McKinsey x ClubMed Theme) ---
st.set_page_config(page_title="ClubMed Executive Intelligence", layout="wide", page_icon="Ψ")

CSS_STYLE = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@300;400;500;600&display=swap');
    :root { --cm-blue: #1D263B; --cm-terracotta: #A64B35; --cm-sage: #A4B6B0; --cm-beige: #F8F9FA; }
    .main { background-color: var(--cm-beige); font-family: 'Inter', sans-serif; }
    h1, h2, h3, h4 { font-family: 'Playfair Display', serif !important; color: var(--cm-blue); }
    div[data-testid="stMetric"] { background-color: white; border-radius: 6px; padding: 15px 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.02); border-top: 3px solid var(--cm-terracotta); }
    div[data-testid="stMetricValue"] { color: var(--cm-blue); font-weight: 600; font-size: 28px; }
    .stDataFrame { border: 1px solid #EAECEF; border-radius: 6px; overflow: hidden; background-color: white; }
    div[data-testid="stChatMessage"] { background-color: transparent !important; border: none !important; border-bottom: 1px solid #EAECEF !important; padding: 1.5rem 0.5rem !important; }
    div[data-testid="stChatMessageAvatarUser"], div[data-testid="stChatMessageAvatarAssistant"] { background-color: var(--cm-blue) !important; color: white !important;}
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

# --- 3. 核心数据清洗 ---
@st.cache_data
def load_and_clean(file):
    data = pd.read_csv(file, low_memory=False)
    data.columns = [col.strip() for col in data.columns]
    mapping = {
        'CONSUMPTION_CALENDAR[Month Name]': 'Month',
        'CONSUMPTION_CALENDAR[Consumption_month_num]': 'Month_Num',
        'CONSUMPTION_CALENDAR[Consumption_year]': 'Year',
        'REF_SALES_MARKET[Market]': 'Market',
        'REF_DESTINATION[Resort]': 'Resort',
        'REF_CML_AGENCY[Group_TA_cml]': 'TA_Group',
        'REF_DESTINATION[Destination type Asia]': 'Dest_Type',
        '[BVSTS___final]': 'BV',
        '[HN_final]': 'HN'
    }
    data.rename(columns=mapping, inplace=True, errors='ignore')
    for col in ['Market', 'Resort', 'TA_Group', 'Dest_Type', 'Month']:
        if col in data.columns: data[col] = data[col].astype(str).str.strip()
    for col in ['BV', 'HN']:
        if col in data.columns: data[col] = pd.to_numeric(data[col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
    data['ADR'] = (data['BV'] / data['HN']).replace([float('inf'), -float('inf')], 0).fillna(0)
    data['Year'] = pd.to_numeric(data['Year'], errors='coerce').fillna(0).astype(int)
    if 'Month_Num' in data.columns: data['Month_Num'] = pd.to_numeric(data['Month_Num'], errors='coerce').fillna(0).astype(int)
    return data

# --- 🌟 高级画图模块 (Plotly) ---
def draw_executive_chart(monthly_df):
    fig = go.Figure()
    # 柱状图：BV
    fig.add_trace(go.Bar(x=monthly_df['Month'], y=monthly_df['BV'], name='Total BV (€)', marker_color='#1D263B', 
                         text=monthly_df['BV'], texttemplate='%{text:,.0f}', textposition='inside'))
    # 柱状图：HN
    fig.add_trace(go.Bar(x=monthly_df['Month'], y=monthly_df['HN'], name='Nights (HN)', marker_color='#A4B6B0', 
                         text=monthly_df['HN'], texttemplate='%{text:,.0f}', textposition='inside'))
    # 折线图：ADR (悬浮在次坐标轴)
    fig.add_trace(go.Scatter(x=monthly_df['Month'], y=monthly_df['ADR'], name='ADR (€)', mode='lines+markers+text', 
                             marker_color='#A64B35', line=dict(width=3), yaxis='y2',
                             text=monthly_df['ADR'], texttemplate='%{text:,.0f}', textposition='top center'))

    fig.update_layout(
        title=dict(text="Monthly Performance Trend", font=dict(family="Playfair Display", size=20, color='#1D263B')),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', margin=dict(t=50, l=0, r=0, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
        yaxis=dict(showgrid=True, gridcolor='#EAECEF', visible=False), # 隐藏坐标轴，依赖直观数据标签
        yaxis2=dict(overlaying='y', side='right', visible=False),
        barmode='group'
    )
    return fig

# --- 🌟 AI 洞察报告生成器 ---
def generate_executive_summary(target, cy_bv, py_bv, summary_df):
    variance = cy_bv - py_bv
    pct = (variance / py_bv * 100) if py_bv > 0 else 0
    sys_prompt = "You are a Senior McKinsey Strategy Consultant working for ClubMed. Write a concise, professional executive summary (max 3 sentences) analyzing this performance. Explain the YoY variance and highlight the top contributing Destination Type. Keep it formal and boardroom-ready."
    user_prompt = f"Target Entity: {target}\nCurrent Year BV: €{cy_bv:,.0f}\nPrevious Year BV: €{py_bv:,.0f}\nYoY Variance: {pct:+.1f}%\n\nPerformance by Destination Type:\n{summary_df.to_string()}"
    
    try:
        resp = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)])
        return resp.content
    except Exception as e:
        return f"The entity generated €{cy_bv:,.0f} in the current period, representing a {pct:+.1f}% YoY change."

# ==========================================
# 🌟 核心拦截提取器
# ==========================================
def extract_payload(resp):
    if isinstance(resp, dict) and resp.get('type') == 'advanced_dashboard': return resp
    if isinstance(resp, pd.DataFrame): return resp
    if isinstance(resp, dict) and 'value' in resp and isinstance(resp['value'], pd.DataFrame): return resp['value']
    if hasattr(resp, 'dataframe'): return resp.dataframe
    try: return pd.DataFrame(resp)
    except: return None

# --- 4. 业务逻辑与界面 ---
with st.sidebar:
    st.markdown("<h2 style='color:#A64B35;'>ClubMed Ψ Hub</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload SalesData.csv", type=['csv'])
    st.divider()

if uploaded_file:
    df = load_and_clean(uploaded_file)
    
    with st.sidebar:
        years = sorted([y for y in df['Year'].unique() if y > 2000], reverse=True)
        sel_year = st.selectbox("Current Year Filter", years) if years else 2026
        markets = sorted([str(m) for m in df['Market'].unique() if str(m).strip() != '' and str(m).lower() != 'nan'])
        sel_markets = st.multiselect("Active Markets (Dashboard)", markets)

    st.markdown(f"### 📈 Executive Performance ({sel_year} vs {sel_year-1})")
    df_cy = df[df['Year'] == sel_year]; df_py = df[df['Year'] == sel_year - 1]
    c1, c2, c3 = st.columns(3)
    c1.metric("Total BV", f"€ {df_cy['BV'].sum():,.0f}", f"{(df_cy['BV'].sum()-df_py['BV'].sum())/df_py['BV'].sum()*100:.1f}%" if df_py['BV'].sum()>0 else None)
    c2.metric("Total HN", f"{df_cy['HN'].sum():,.0f}")
    c3.metric("Avg ADR", f"€ {df_cy['BV'].sum()/df_cy['HN'].sum() if df_cy['HN'].sum()>0 else 0:,.2f}")

    # ==========================================
    # 🌟 模块 C：全能型战略 AI 顾问
    # ==========================================
    st.divider()
    st.markdown("### 🤖 Strategy Advisor (Full Dashboard Engine)")
    
    if "messages" not in st.session_state: st.session_state.messages = []

    agent = Agent(df, config={"llm": llm, "custom_instructions": "You generate strictly valid Python code.", "save_charts": False, "enable_cache": False})

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            if isinstance(m["content"], str): st.markdown(m["content"])

    if prompt := st.chat_input("E.g., Breakdown China Mainland performance from Jan to May 2026."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.write(prompt)
            
        hacked_prompt = f"""
        User Question: {prompt}
        
        [SYSTEM OVERRIDE]: 
        DO NOT GENERATE ANY PLOTS! YOU MUST EXECUTE THIS EXACT CODE STRUCTURE TO RETURN A DICTIONARY PAYLOAD:

        ```python
        import pandas as pd
        import numpy as np

        df_base = dfs[0].copy()
        
        # 1. Target Filtering (Adapt TARGET_ENTITY and CORRECT_COLUMN from: Market, TA_Group, Resort, Dest_Type)
        df_target = df_base[df_base['CORRECT_COLUMN'].str.contains('TARGET_ENTITY', case=False, na=False)]
        
        # 2. Time Filtering (CY = 2026, PY = 2025)
        df_cy = df_target[(df_target['Year'] == 2026) & (df_target['Month_Num'].between(1, 5))]
        df_py = df_target[(df_target['Year'] == 2025) & (df_target['Month_Num'].between(1, 5))]

        # 3. Table 1: Summary by Dest_Type ONLY
        summary_df = df_cy.groupby('Dest_Type').agg({{'BV': 'sum', 'HN': 'sum'}}).reset_index()
        summary_df['ADR'] = np.where(summary_df['HN'] > 0, summary_df['BV'] / summary_df['HN'], 0)

        # 4. Table 2: Monthly Breakdown ONLY
        monthly_df = df_cy.groupby(['Month_Num', 'Month']).agg({{'BV': 'sum', 'HN': 'sum'}}).reset_index().sort_values('Month_Num')
        monthly_df['ADR'] = np.where(monthly_df['HN'] > 0, monthly_df['BV'] / monthly_df['HN'], 0)
        monthly_df = monthly_df.drop(columns=['Month_Num'])

        # 5. Output Payload
        result = {{
            'type': 'advanced_dashboard',
            'summary': summary_df,
            'monthly': monthly_df,
            'cy_bv': float(df_cy['BV'].sum()),
            'py_bv': float(df_py['BV'].sum()),
            'target': 'TARGET_ENTITY'
        }}
        ```
        CRITICAL: ASSIGN THE DICTIONARY EXACTLY AS SHOWN TO THE VARIABLE `result`.
        """

        with st.chat_message("assistant"):
            with st.spinner("Extracting multi-dimensional data & compiling McKinsey insights..."):
                try:
                    response = agent.chat(hacked_prompt)
                    payload = extract_payload(response)
                    
                    if isinstance(payload, dict) and payload.get('type') == 'advanced_dashboard':
                        st.markdown("**1️⃣ Performance by Destination Type (Total)**")
                        summary_style = payload['summary'].style.format({'BV': '€ {:,.0f}', 'HN': '{:,.0f}', 'ADR': '€ {:,.0f}'})
                        st.dataframe(summary_style, use_container_width=True, hide_index=True)
                        
                        st.markdown("**2️⃣ Monthly Performance Trend**")
                        st.plotly_chart(draw_executive_chart(payload['monthly']), use_container_width=True)
                        
                        insight_text = generate_executive_summary(payload['target'], payload['cy_bv'], payload['py_bv'], payload['summary'])
                        st.info(f"💡 **Executive Insight:**\n\n{insight_text}")
                        
                        st.session_state.messages.append({"role": "assistant", "content": f"✅ Dashboard for **{payload['target']}** generated successfully."})
                        
                    elif isinstance(payload, pd.DataFrame):
                        st.dataframe(payload, use_container_width=True)
                        st.session_state.messages.append({"role": "assistant", "content": "Here is the data table."})
                    else:
                        st.warning("⚠️ Data parsing failed. See code below.")
                        
                    code_executed = getattr(agent, 'last_code_executed', None)
                    if code_executed:
                        with st.expander("🛠️ View Architecture Code"): st.code(code_executed, language='python')
                            
                except Exception as e:
                    st.error(f"Execution Error: {e}")
else:
    st.markdown("""
        <div style="height: 60vh; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center;">
            <h1 style="font-size: 3rem; margin-bottom: 1rem; color: #1D263B;">Ψ Executive Strategy Hub</h1>
            <p style="color: #6c757d; max-width: 550px; font-size: 1.1rem; line-height: 1.6;">Upload data to unlock Plotly Dashboards and AI Executive Insights.</p>
        </div>
    """, unsafe_allow_html=True)
