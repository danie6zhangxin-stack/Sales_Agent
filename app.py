import streamlit as st
import pandas as pd
import numpy as np 
from pandasai import Agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import plotly.graph_objects as go
import datetime
import os

# --- 1. 高端商业视觉配置 ---
st.set_page_config(page_title="ClubMed Executive Intelligence", layout="wide", page_icon="Ψ")

CSS_STYLE = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@300;400;500;600&display=swap');
    :root { --cm-blue: #1D263B; --cm-terracotta: #A64B35; --cm-sage: #A4B6B0; --cm-beige: #F8F9FA; }
    .main { background-color: var(--cm-beige); font-family: 'Inter', sans-serif; }
    h1, h2, h3, h4 { font-family: 'Playfair Display', serif !important; color: var(--cm-blue); }
    div[data-testid="stMetric"] { background-color: white; border-radius: 6px; padding: 15px 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.02); border-top: 3px solid var(--cm-terracotta); border-left: 4px solid var(--cm-blue); }
    .stDataFrame { border: 1px solid #EAECEF; border-radius: 6px; overflow: hidden; background-color: white; }
    .stSidebar { background-color: white !important; border-right: 1px solid #EAECEF; }
</style>
"""
st.markdown(CSS_STYLE, unsafe_allow_html=True)

# --- 2. AI 引擎初始化 ---
try:
    api_key = st.secrets["DEEPSEEK_API_KEY"]
except:
    api_key = "sk-xxxxxxxxxxxxxxxx" 

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
        'REF_SALES_MARKET[Market]': 'Market',
        'REF_DESTINATION[Resort]': 'Resort',
        'REF_CML_AGENCY[Group_TA_cml]': 'TA_Group',
        'REF_DESTINATION[Destination type Asia]': 'Dest_Type',
        '[BVSTS___final]': 'BV',
        '[HN_final]': 'HN'
    }
    data.rename(columns=mapping, inplace=True, errors='ignore')
    
    # 基础清洗
    for col in ['Market', 'Resort', 'TA_Group', 'Dest_Type', 'Month']:
        if col in data.columns: data[col] = data[col].astype(str).str.strip()
    for col in ['BV', 'HN']:
        if col in data.columns: data[col] = pd.to_numeric(data[col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
    
    data['Year'] = pd.to_numeric(data['Year'], errors='coerce').fillna(0).astype(int)
    data['Month_Num'] = pd.to_numeric(data['Month_Num'], errors='coerce').fillna(0).astype(int)
    if 'Sales_Date' in data.columns:
        data['Sales_Date'] = pd.to_datetime(data['Sales_Date'], errors='coerce')
        
    return data

# --- 🌟 高级画图模块 (Pacing Comparison) ---
def draw_pacing_chart(cy_df, py_df, cy_label, py_label):
    cy_g = cy_df.groupby('Dest_Type')[['BV']].sum().reset_index()
    py_g = py_df.groupby('Dest_Type')[['BV']].sum().reset_index()
    combined = pd.merge(cy_g, py_g, on='Dest_Type', how='outer', suffixes=('_CY', '_PY')).fillna(0)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=combined['Dest_Type'], y=combined['BV_CY'], name=cy_label, marker_color='#1D263B', text=combined['BV_CY'], texttemplate='%{text:,.0f}', textposition='auto'))
    fig.add_trace(go.Bar(x=combined['Dest_Type'], y=combined['BV_PY'], name=py_label, marker_color='#A4B6B0', text=combined['BV_PY'], texttemplate='%{text:,.0f}', textposition='auto'))
    
    fig.update_layout(barmode='group', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', margin=dict(t=20, b=0, l=0, r=0), legend=dict(orientation="h", y=1.1, x=0.5, xanchor='center'))
    return fig

# --- 🌟 AI 逻辑提取器 ---
def extract_payload(resp):
    if isinstance(resp, dict) and resp.get('type') == 'advanced_dashboard': return resp
    if isinstance(resp, pd.DataFrame): return resp
    if isinstance(resp, dict) and 'value' in resp and isinstance(resp['value'], pd.DataFrame): return resp['value']
    try: return pd.DataFrame(resp)
    except: return None

# --- 4. 业务界面 ---
if uploaded_file := st.sidebar.file_uploader("Upload SalesData.csv", type=['csv']):
    df = load_and_clean(uploaded_file)
    
    with st.sidebar:
        st.markdown("### 🛠️ Global Filters")
        
        # 1. Consumption Year
        sel_year = st.selectbox("Consumption Year", sorted(df['Year'].unique(), reverse=True), index=0)
        
        # 2. Season Selection (S1/S2 修正版)
        # S1: Jan-Jun (1-6), S2: Jul-Dec (7-12)
        season = st.radio("Season Focus", ["All Year", "S1 (Jan-Jun)", "S2 (Jul-Dec)"])
        
        # 3. Market Selection
        all_markets = sorted(df['Market'].unique())
        sel_markets = st.multiselect("Market Select", all_markets, default=[m for m in all_markets if "China" in m or "Hong Kong" in m])
        
        # 4. TA Selection
        all_tas = sorted(df['TA_Group'].unique())
        sel_ta = st.multiselect("Travel Agency Select", all_tas)

        st.divider()
        st.markdown("### 📅 Sales Date Range (Pacing)")
        
        # 销售日期区间选择
        max_date = df['Sales_Date'].max().date() if not df['Sales_Date'].dropna().empty else datetime.date.today()
        start_default = max_date - datetime.timedelta(days=90)
        date_range = st.date_input("Booking Window:", value=(start_default, max_date))
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            py_start = start_date.replace(year=start_date.year - 1)
            py_end = end_date.replace(year=end_date.year - 1)
            st.caption(f"Comparing CY ({start_date} to {end_date}) \nvs PY ({py_start} to {py_end})")
        else:
            st.warning("Please select a complete range.")
            st.stop()

    # --- 🌟 核心过滤逻辑 ---
    def apply_filters(input_df, year_val, s_date, e_date):
        d = input_df[input_df['Year'] == year_val]
        # 时间区间过滤
        d = d[(d['Sales_Date'].dt.date >= s_date) & (d['Sales_Date'].dt.date <= e_date)]
        
        # 🌟 Season 过滤 (修正为 1-6月 和 7-12月)
        if season == "S1 (Jan-Jun)": 
            d = d[d['Month_Num'].between(1, 6)]
        elif season == "S2 (Jul-Dec)": 
            d = d[d['Month_Num'].between(7, 12)]
            
        # Market & TA 过滤
        if sel_markets: d = d[d['Market'].isin(sel_markets)]
        if sel_ta: d = d[d['TA_Group'].isin(sel_ta)]
        return d

    df_cy_base = apply_filters(df, sel_year, start_date, end_date)
    df_py_base = apply_filters(df, sel_year - 1, py_start, py_end)

    # --- KPI 展示 ---
    st.markdown(f"### 📊 Booking Pacing Dashboard: {sel_year} vs {sel_year-1}")
    c1, c2, c3 = st.columns(3)
    cy_bv = df_cy_base['BV'].sum(); py_bv = df_py_base['BV'].sum()
    c1.metric("Paced BV", f"€ {cy_bv:,.0f}", f"{(cy_bv-py_bv)/py_bv*100:.1f}%" if py_bv>0 else None)
    c2.metric("Paced HN", f"{df_cy_base['HN'].sum():,.0f}")
    c3.metric("Current ADR", f"€ {df_cy_base['BV'].sum()/df_cy_base['HN'].sum() if df_cy_base['HN'].sum()>0 else 0:,.2f}")

    # --- Pacing 图表 ---
    st.plotly_chart(draw_pacing_chart(df_cy_base, df_py_base, f"CY ({sel_year})", f"PY ({sel_year-1})"), use_container_width=True)

    # --- 🤖 AI Strategy Advisor ---
    st.divider()
    st.markdown("### 🤖 Strategy Advisor")
    st.caption(f"Current Context: **{season}** | Market: **{', '.join(sel_markets) if sel_markets else 'All'}** | Booking Window: **{start_date} to {end_date}**")

    if "messages" not in st.session_state: st.session_state.messages = []
    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.markdown(m["content"])

    if prompt := st.chat_input("Explain the variance or deep dive into specific resorts."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.write(prompt)
            
        hacked_prompt = f"""
        User Question: {prompt}
        
        [CONTEXT OVERRIDE]:
        The user has already selected the following filters in the UI:
        - Target Year: {sel_year}
        - Season: {season}
        - Markets: {sel_markets}
        - TAs: {sel_ta}
        - Booking Window CY: {start_date} to {end_date}
        - Booking Window PY: {py_start} to {py_end}

        [SYSTEM TASK]:
        Generate Python code to create a 'advanced_dashboard' payload.
        1. Apply the EXACT same filters mentioned above.
        2. Use normalized fuzzy search (uppercase + strip spaces) for any additional names mentioned in user question.
        3. result must be a dictionary: {{'type': 'advanced_dashboard', 'summary': df, 'monthly': df_trend, 'cy_bv': val, 'py_bv': val, 'target': 'Contextual Summary'}}
        """

        agent = Agent(df, config={"llm": llm, "save_charts": False})
        with st.chat_message("assistant"):
            with st.spinner("Analyzing current pacing context..."):
                try:
                    response = agent.chat(hacked_prompt)
                    payload = extract_payload(response)
                    
                    if isinstance(payload, dict) and payload.get('type') == 'advanced_dashboard':
                        st.dataframe(payload['summary'], use_container_width=True, hide_index=True)
                        st.session_state.messages.append({"role": "assistant", "content": "Analysis complete based on your active filters."})
                    else:
                        st.markdown(response)
                except Exception as e:
                    st.error(f"Error: {e}")
else:
    st.info("Please upload your SalesData.csv to start the Pacing Analysis.")
