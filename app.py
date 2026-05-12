import streamlit as st
import pandas as pd
import numpy as np 
from pandasai import Agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import plotly.graph_objects as go
import datetime
import os

# --- 1. 高端商业视觉配置 (McKinsey x ClubMed Theme) ---
st.set_page_config(page_title="ClubMed Executive Intelligence", layout="wide", page_icon="Ψ")

CSS_STYLE = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@300;400;500;600&display=swap');
    :root { --cm-blue: #1D263B; --cm-terracotta: #A64B35; --cm-sage: #A4B6B0; --cm-beige: #F8F9FA; }
    .main { background-color: var(--cm-beige); font-family: 'Inter', sans-serif; }
    h1, h2, h3, h4 { font-family: 'Playfair Display', serif !important; color: var(--cm-blue); }
    div[data-testid="stMetric"] { background-color: white; border-radius: 6px; padding: 15px 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.02); border-top: 3px solid var(--cm-terracotta); border-left: 4px solid var(--cm-blue); }
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
    api_key = "sk-xxxxxxxxxxxxxxxxxxxxxxxx" 

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
        'SALES_CALENDAR[Sales_date]': 'Sales_Date',
        'SALES_CALENDAR[Sales_Month_name]': 'Sales_Month',
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
    
    # 🌟 日期格式化，核心处理
    if 'Sales_Date' in data.columns:
        data['Sales_Date'] = pd.to_datetime(data['Sales_Date'], errors='coerce')
    if 'Month_Num' in data.columns: 
        data['Month_Num'] = pd.to_numeric(data['Month_Num'], errors='coerce').fillna(0).astype(int)
        
    return data

# --- 🌟 高级画图模块 (AI洞察与趋势) ---
def draw_executive_chart(monthly_df):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=monthly_df['Month'], y=monthly_df['BV'], name='Total BV (€)', marker_color='#1D263B', text=monthly_df['BV'], texttemplate='%{text:,.0f}', textposition='inside'))
    fig.add_trace(go.Bar(x=monthly_df['Month'], y=monthly_df['HN'], name='Nights (HN)', marker_color='#A4B6B0', text=monthly_df['HN'], texttemplate='%{text:,.0f}', textposition='inside'))
    fig.add_trace(go.Scatter(x=monthly_df['Month'], y=monthly_df['ADR'], name='ADR (€)', mode='lines+markers+text', marker_color='#A64B35', line=dict(width=3), yaxis='y2', text=monthly_df['ADR'], texttemplate='%{text:,.0f}', textposition='top center'))
    fig.update_layout(title=dict(text="Monthly Performance Trend", font=dict(family="Playfair Display", size=20)), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', legend=dict(orientation="h", y=1.1, x=1), yaxis=dict(visible=False), yaxis2=dict(overlaying='y', side='right', visible=False), barmode='group')
    return fig

def generate_executive_summary(target, cy_bv, py_bv, summary_df):
    variance = cy_bv - py_bv
    pct = (variance / py_bv * 100) if py_bv > 0 else 0
    sys_prompt = "You are a Senior McKinsey Consultant. Write a 3-sentence formal analysis focusing on YoY variance and key segments. Use business English."
    user_prompt = f"Entity: {target}\nCY BV: €{cy_bv:,.0f}\nPY BV: €{py_bv:,.0f}\nVariance: {pct:+.1f}%\nSummary:\n{summary_df.to_string()}"
    try:
        resp = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)])
        return resp.content
    except:
        return f"Performance for {target} shows a {pct:+.1f}% YoY change with total BV at €{cy_bv:,.0f}."

def extract_payload(resp):
    if isinstance(resp, dict) and resp.get('type') == 'advanced_dashboard': return resp
    if isinstance(resp, pd.DataFrame): return resp
    if isinstance(resp, dict) and 'value' in resp and isinstance(resp['value'], pd.DataFrame): return resp['value']
    if hasattr(resp, 'dataframe'): return resp.dataframe
    try: return pd.DataFrame(resp)
    except: return None

# --- 4. 业务逻辑与界面 ---
if uploaded_file := st.sidebar.file_uploader("Upload SalesData.csv", type=['csv']):
    df = load_and_clean(uploaded_file)
    
    with st.sidebar:
        st.markdown("### ⚙️ Global Constraints")
        years = sorted([y for y in df['Year'].unique() if y > 2000], reverse=True)
        sel_year = st.selectbox("Consumption Year", years) if years else 2026
        
        # 🌟 核心升级：销售日期 Pacing 筛选器
        st.markdown("### 📅 Booking Pace (STLY)")
        valid_dates = df['Sales_Date'].dropna()
        default_date = valid_dates.max().date() if not valid_dates.empty else datetime.date.today()
        as_of_date = st.date_input("Sales Booked As Of:", value=default_date)
        
        # 计算去年同期的比较日期
        try:
            py_as_of_date = as_of_date.replace(year=as_of_date.year - 1)
        except ValueError: # 处理闰年 2月29日
            py_as_of_date = as_of_date.replace(year=as_of_date.year - 1, day=28)
            
        st.caption(f"*Comparing vs equivalent booking date: **{py_as_of_date.strftime('%Y-%m-%d')}***")

        st.divider()
        st.markdown("### 🔍 Data Detective")
        search_term = st.text_input("Database Name Lookup:")
        if search_term:
            for col in ['TA_Group', 'Market', 'Resort']:
                m = df[df[col].str.contains(search_term, case=False, na=False)][col].unique()
                if len(m) > 0:
                    st.write(f"🏷️ {col}:")
                    for name in m[:5]: st.code(name)

    # 🌟 按照同频的 Pacing 时间轴过滤数据
    df_cy_base = df[(df['Year'] == sel_year) & (df['Sales_Date'].dt.date <= as_of_date)]
    df_py_base = df[(df['Year'] == sel_year - 1) & (df['Sales_Date'].dt.date <= py_as_of_date)]

    st.markdown(f"### 📈 Executive Booking Pace ({sel_year} vs {sel_year-1})")
    st.markdown(f"<p style='color:#6c757d; font-size: 14px; margin-top:-10px;'>Performance booked strictly as of <b>{as_of_date.strftime('%d %b')}</b></p>", unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    cy_total_bv = df_cy_base['BV'].sum()
    py_total_bv = df_py_base['BV'].sum()
    c1.metric("Paced BV", f"€ {cy_total_bv:,.0f}", f"{(cy_total_bv-py_total_bv)/py_total_bv*100:.1f}%" if py_total_bv>0 else None)
    c2.metric("Paced HN", f"{df_cy_base['HN'].sum():,.0f}")
    c3.metric("Avg ADR", f"€ {df_cy_base['BV'].sum()/df_cy_base['HN'].sum() if df_cy_base['HN'].sum()>0 else 0:,.2f}")

    # 🌟 全新 Pacing 分类对比图表
    st.markdown(f"#### 📊 Booking Pace by Destination Type")
    cy_dest = df_cy_base.groupby('Dest_Type')[['BV']].sum().reset_index()
    py_dest = df_py_base.groupby('Dest_Type')[['BV']].sum().reset_index()
    dash_df = pd.merge(cy_dest, py_dest, on='Dest_Type', how='outer', suffixes=(f'_{sel_year}', f'_{sel_year-1}')).fillna(0)
    
    if not dash_df.empty:
        fig_pacing = go.Figure()
        fig_pacing.add_trace(go.Bar(
            x=dash_df['Dest_Type'], y=dash_df[f'BV_{sel_year}'], 
            name=f'{sel_year} (As of {as_of_date.strftime("%d %b")})', 
            marker_color='#1D263B', text=dash_df[f'BV_{sel_year}'], texttemplate='%{text:,.0f}', textposition='auto'
        ))
        fig_pacing.add_trace(go.Bar(
            x=dash_df['Dest_Type'], y=dash_df[f'BV_{sel_year-1}'], 
            name=f'{sel_year-1} (As of {py_as_of_date.strftime("%d %b")})', 
            marker_color='#A4B6B0', text=dash_df[f'BV_{sel_year-1}'], texttemplate='%{text:,.0f}', textposition='auto'
        ))
        fig_pacing.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', barmode='group',
            yaxis=dict(visible=False), legend=dict(orientation="h", y=1.1, x=0.5, xanchor='center'),
            margin=dict(t=20, b=0, l=0, r=0)
        )
        st.plotly_chart(fig_pacing, use_container_width=True)
    else:
        st.info("No pacing data available for the selected dates.")

    # ==========================================
    # 🌟 模块 C：全能型战略 AI 顾问 (保留最强大的分析能力)
    # ==========================================
    st.divider()
    st.markdown("### 🤖 Strategy Advisor (Deep Dive Engine)")
    if "messages" not in st.session_state: st.session_state.messages = []
    
    agent = Agent(df, config={"llm": llm, "custom_instructions": "Generate strictly valid Python code.", "save_charts": False, "enable_cache": False})

    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.markdown(m["content"])

    if prompt := st.chat_input("E.g., Analyze performance for NJ XXY."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.write(prompt)
            
        hacked_prompt = f"""
        User Question: {prompt}
        
        [SYSTEM OVERRIDE - ROBUST MATCHING]: 
        1. STRATEGY: Use a normalized 'Omni-search' approach.
        2. NORMALIZATION: Strip ALL spaces and convert to UPPERCASE for both the user-provided name and the database columns.
        3. CODE STRUCTURE:

        ```python
        import pandas as pd
        import numpy as np

        df_base = dfs[0].copy()
        raw_target = 'EXTRACT_TARGET_NAME_HERE' 
        clean_target = raw_target.replace(' ', '').upper()
        
        mask = (
            df_base['Market'].str.replace(' ', '', regex=False).str.upper().str.contains(clean_target, na=False) |
            df_base['TA_Group'].str.replace(' ', '', regex=False).str.upper().str.contains(clean_target, na=False) |
            df_base['Resort'].str.replace(' ', '', regex=False).str.upper().str.contains(clean_target, na=False)
        )
        df_target = df_base[mask]
        
        df_cy = df_target[(df_target['Year'] == 2026) & (df_target['Month_Num'].between(1, 5))]
        df_py = df_target[(df_target['Year'] == 2025) & (df_target['Month_Num'].between(1, 5))]

        summary_df = df_cy.groupby('Dest_Type').agg({{'BV': 'sum', 'HN': 'sum'}}).reset_index()
        summary_df['ADR'] = np.where(summary_df['HN'] > 0, summary_df['BV'] / summary_df['HN'], 0)

        monthly_df = df_cy.groupby(['Month_Num', 'Month']).agg({{'BV': 'sum', 'HN': 'sum'}}).reset_index().sort_values('Month_Num')
        monthly_df['ADR'] = np.where(monthly_df['HN'] > 0, monthly_df['BV'] / monthly_df['HN'], 0)
        monthly_df = monthly_df.drop(columns=['Month_Num'])

        result = {{
            'type': 'advanced_dashboard',
            'summary': summary_df, 'monthly': monthly_df,
            'cy_bv': float(df_cy['BV'].sum()), 'py_bv': float(df_py['BV'].sum()), 'target': raw_target
        }}
        ```
        """

        with st.chat_message("assistant"):
            with st.spinner("Executing normalized fuzzy matching & fetching insights..."):
                try:
                    response = agent.chat(hacked_prompt)
                    payload = extract_payload(response)
                    
                    if isinstance(payload, dict) and payload.get('type') == 'advanced_dashboard':
                        st.markdown(f"#### 📊 Executive Report: {payload['target']}")
                        st.dataframe(payload['summary'].style.format({'BV': '€ {:,.0f}', 'HN': '{:,.0f}', 'ADR': '€ {:,.0f}'}), use_container_width=True, hide_index=True)
                        st.plotly_chart(draw_executive_chart(payload['monthly']), use_container_width=True)
                        st.info(f"💡 **McKinsey Insight:**\n\n{generate_executive_summary(payload['target'], payload['cy_bv'], payload['py_bv'], payload['summary'])}")
                        st.session_state.messages.append({"role": "assistant", "content": f"Dashboard for {payload['target']} ready."})
                    else:
                        st.write(response)
                except Exception as e:
                    st.error(f"Error: {e}")
else:
    st.markdown("<h1 style='text-align:center; padding-top:150px; opacity:0.1;'>Ψ Executive Hub</h1>", unsafe_allow_html=True)
