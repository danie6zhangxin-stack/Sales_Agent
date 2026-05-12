import streamlit as st
import pandas as pd
import numpy as np 
from pandasai import Agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import plotly.graph_objects as go
import datetime

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
    div[data-testid="stChatMessage"] { background-color: transparent !important; border: none !important; border-bottom: 1px solid #EAECEF !important; padding: 1.5rem 0.5rem !important; }
    div[data-testid="stChatMessageAvatarUser"], div[data-testid="stChatMessageAvatarAssistant"] { background-color: var(--cm-blue) !important; color: white !important; }
    div[data-testid="stChatMessageAvatarUser"] svg, div[data-testid="stChatMessageAvatarAssistant"] svg { fill: white !important; color: white !important; }
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
    for col in ['Market', 'Resort', 'TA_Group', 'Dest_Type', 'Month']:
        if col in data.columns: data[col] = data[col].astype(str).str.strip()
    for col in ['BV', 'HN']:
        if col in data.columns: data[col] = pd.to_numeric(data[col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
    data['Year'] = pd.to_numeric(data['Year'], errors='coerce').fillna(0).astype(int)
    data['Month_Num'] = pd.to_numeric(data['Month_Num'], errors='coerce').fillna(0).astype(int)
    if 'Sales_Date' in data.columns:
        data['Sales_Date'] = pd.to_datetime(data['Sales_Date'], errors='coerce')
    return data

# --- 🌟 高级画图模块 ---
def draw_pacing_chart(cy_df, py_df, cy_label, py_label, dynamic_title):
    cy_g = cy_df.groupby('Dest_Type')[['BV']].sum().reset_index()
    py_g = py_df.groupby('Dest_Type')[['BV']].sum().reset_index()
    
    cy_g['BV'] = cy_g['BV'] / 1000
    py_g['BV'] = py_g['BV'] / 1000
    
    combined = pd.merge(cy_g, py_g, on='Dest_Type', how='outer', suffixes=('_CY', '_PY')).fillna(0)
    combined['YoY_Pct'] = np.where(combined['BV_PY'] > 0, (combined['BV_CY'] - combined['BV_PY']) / combined['BV_PY'] * 100, 0)
    
    text_cy = [f"<b>{cy:,.0f}k<br>({pct:+.1f}%)</b>" if py > 0 else f"<b>{cy:,.0f}k</b>" for cy, py, pct in zip(combined['BV_CY'], combined['BV_PY'], combined['YoY_Pct'])]
    text_py = [f"<b>{py:,.0f}k</b>" for py in combined['BV_PY']]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=combined['Dest_Type'], y=combined['BV_CY'], name=cy_label, marker_color='#1D263B', text=text_cy, textposition='auto', textfont=dict(size=14)))
    fig.add_trace(go.Bar(x=combined['Dest_Type'], y=combined['BV_PY'], name=py_label, marker_color='#A4B6B0', text=text_py, textposition='auto', textfont=dict(size=14)))
    
    fig.update_layout(title=dict(text=dynamic_title, font=dict(family="Playfair Display", size=18, color='#1D263B'), y=0.95),
                      barmode='group', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', margin=dict(t=70, b=0, l=0, r=0), 
                      legend=dict(orientation="h", y=1.05, x=0.5, xanchor='center'), yaxis=dict(visible=False))
    return fig

def draw_horizontal_bar(data_df, group_col, title, color):
    g_df = data_df.groupby(group_col)['BV'].sum().reset_index().sort_values('BV', ascending=True).tail(5)
    g_df['BV'] = g_df['BV'] / 1000
    
    fig = go.Figure(go.Bar(x=g_df['BV'], y=g_df[group_col], orientation='h', marker_color=color, text=g_df['BV'], texttemplate='<b>%{text:,.0f}k</b>', textposition='inside', textfont=dict(size=12, color='white')))
    fig.update_layout(title=dict(text=title, font=dict(family="Playfair Display", size=16)), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', margin=dict(t=40, b=0, l=0, r=0), xaxis=dict(visible=False), yaxis=dict(showgrid=False))
    return fig

# --- 🌟 AI 宏观经济战略洞察生成器 ---
def generate_macro_insights(cy_data, py_data, context_desc):
    cy_bv = cy_data['BV'].sum() / 1000 
    py_bv = py_data['BV'].sum() / 1000
    pct = ((cy_bv - py_bv) / py_bv * 100) if py_bv > 0 else 0
    
    top_resorts = cy_data.groupby('Resort')['BV'].sum().nlargest(3).to_dict()
    top_tas = cy_data.groupby('TA_Group')['BV'].sum().nlargest(3).to_dict()
    
    sys_prompt = """You are the Chief Strategy Officer and Macro-Economist for ClubMed. 
    Analyze the YoY pacing variance.
    
    CRITICAL STRUCTURE FOR YOUR RESPONSE:
    1. **Macro-Environmental Shift**: Explain the variance NOT just with numbers, but by hypothesizing plausible MACRO factors relevant to the 2025/2026 global landscape (e.g., shifting geopolitical tensions, economic policies, visa-free travel, flight capacity, currency fluctuations).
    2. **Strategic Pivot**: Explain how these macro events force a shift in consumer behavior (e.g., flight to safety, booking window changes).
    3. **Micro-Execution**: Briefly tie the theory to the top performing Resorts and TAs provided in the data.
    
    Write a compelling, boardroom-ready story (4-5 sentences max)."""
    
    user_prompt = f"UI Context: {context_desc}\nCY Total: {cy_bv:,.0f} k€ | PY Total: {py_bv:,.0f} k€ | Variance: {pct:+.1f}%\n\nTop Resorts CY: {top_resorts}\nTop TAs CY: {top_tas}"
    
    try:
        resp = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)])
        return resp.content
    except:
        return "Unable to generate macro insights at this time."

def extract_dataframe(resp):
    if isinstance(resp, pd.DataFrame): return resp
    if isinstance(resp, dict) and 'value' in resp and isinstance(resp['value'], pd.DataFrame): return resp['value']
    if hasattr(resp, 'dataframe'): return resp.dataframe
    try: return pd.DataFrame(resp)
    except: return None

# ==========================================
# 🌟 4. 业务界面逻辑
# ==========================================
if uploaded_file := st.sidebar.file_uploader("Upload SalesData.csv", type=['csv']):
    df = load_and_clean(uploaded_file)
    
    with st.sidebar:
        st.markdown("### 🛠️ Global Filters")
        sel_year = st.selectbox("Consumption Year", sorted(df['Year'].unique(), reverse=True), index=0)
        season = st.radio("Season Focus", ["All Year", "S1 (Jan-Jun)", "S2 (Jul-Dec)"])
        sel_markets = st.multiselect("Market Select", sorted(df['Market'].unique()))
        sel_ta = st.multiselect("Travel Agency Select", sorted(df['TA_Group'].unique()))

        st.divider()
        st.markdown("### 📅 Booking Window (Pacing)")
        max_date = df['Sales_Date'].max().date() if not df['Sales_Date'].dropna().empty else datetime.date.today()
        
        preset = st.selectbox("Quick Range Select", ["Last 3 Months", "Last Week", "Last 1 Month", "Last 6 Months", "Last 1 Year", "Custom Range"])
        
        if preset == "Custom Range":
            col_start, col_end = st.columns(2)
            start_date = col_start.date_input("Start Date", value=max_date - datetime.timedelta(days=90))
            end_date = col_end.date_input("End Date", value=max_date)
        else:
            if preset == "Last Week": start_date = max_date - datetime.timedelta(days=7)
            elif preset == "Last 1 Month": start_date = max_date - datetime.timedelta(days=30)
            elif preset == "Last 3 Months": start_date = max_date - datetime.timedelta(days=90)
            elif preset == "Last 6 Months": start_date = max_date - datetime.timedelta(days=180)
            elif preset == "Last 1 Year": start_date = max_date - datetime.timedelta(days=365)
            end_date = max_date
            st.info(f"📅 Active Window:\n**{start_date.strftime('%d %b %Y')}** to **{end_date.strftime('%d %b %Y')}**")
        
        if start_date <= end_date:
            try:
                py_start, py_end = start_date.replace(year=start_date.year-1), end_date.replace(year=end_date.year-1)
            except ValueError:
                py_start, py_end = start_date - datetime.timedelta(days=365), end_date - datetime.timedelta(days=365)
        else:
            st.error("Start Date must be before End Date.")
            st.stop()

    def apply_ui_filters(input_df, year_val, s_date, e_date):
        d = input_df[input_df['Year'] == year_val]
        d = d[(d['Sales_Date'].dt.date >= s_date) & (d['Sales_Date'].dt.date <= e_date)]
        if season == "S1 (Jan-Jun)": d = d[d['Month_Num'].between(1, 6)]
        elif season == "S2 (Jul-Dec)": d = d[d['Month_Num'].between(7, 12)]
        if sel_markets: d = d[d['Market'].isin(sel_markets)]
        if sel_ta: d = d[d['TA_Group'].isin(sel_ta)]
        return d

    df_cy_base = apply_ui_filters(df, sel_year, start_date, end_date)
    df_py_base = apply_ui_filters(df, sel_year - 1, py_start, py_end)

    st.markdown(f"### 📈 Executive Booking Pacing: {sel_year} vs {sel_year-1}")
    
    cy_bv = df_cy_base['BV'].sum() / 1000
    py_bv = df_py_base['BV'].sum() / 1000
    cy_hn = df_cy_base['HN'].sum()
    py_hn = df_py_base['HN'].sum()
    cy_adr = (cy_bv * 1000) / cy_hn if cy_hn > 0 else 0
    py_adr = (py_bv * 1000) / py_hn if py_hn > 0 else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Paced BV (k€)", f"k€ {cy_bv:,.0f}", f"{(cy_bv-py_bv)/py_bv*100:.1f}%" if py_bv>0 else None)
    c2.metric("Paced HN", f"{cy_hn:,.0f}", f"{(cy_hn-py_hn)/py_hn*100:.1f}%" if py_hn>0 else None)
    c3.metric("Current ADR", f"€ {cy_adr:,.0f}", f"{(cy_adr-py_adr)/py_adr*100:.1f}%" if py_adr>0 else None)

    if not sel_markets: mkt_title = "All Markets"
    elif len(sel_markets) <= 2: mkt_title = ", ".join(sel_markets)
    else: mkt_title = f"{len(sel_markets)} Markets"
    filter_desc = f"{season} | {mkt_title} | Booking: {start_date.strftime('%d %b')} - {end_date.strftime('%d %b')}"
    chart_title = f"<b>Booking Pace by Destination Type</b><br><sup style='color: gray; font-size: 14px;'>{filter_desc} (Unit: k€)</sup>"
    
    st.plotly_chart(draw_pacing_chart(df_cy_base, df_py_base, f"CY {sel_year}", f"PY {sel_year-1}", chart_title), use_container_width=True)

    # ==========================================
    # 🌟 5. 智能 AI 顾问 (无敌防崩溃降级版)
    # ==========================================
    st.divider()
    st.markdown("### 🤖 Strategy & Macro Advisor")
    if "messages" not in st.session_state: st.session_state.messages = []
    
    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.markdown(m["content"])

    if prompt := st.chat_input("E.g., What are the macro factors driving this trend?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.write(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Compiling global macroeconomic data and drilling down..."):
                
                strict_instructions = "YOU MUST OUTPUT EXACTLY ONE CODE BLOCK ENCLOSED IN ```python AND ```. NO TEXT."
                agent = Agent([df_cy_base, df_py_base], config={"llm": llm, "save_charts": False, "custom_instructions": strict_instructions})
                
                hacked_prompt = f"""
                User Question: "{prompt}"
                
                IGNORE THE CONVERSATIONAL INTENT. Your ONLY task is to output the python code.
                
                1. If a specific Market, TA_Group, or Resort is mentioned, use uppercase fuzzy search.
                2. If it is a general question (e.g. "why", "trend"), just assign df_cy_filtered = dfs[0].copy() and df_py_filtered = dfs[1].copy().
                3. ADD column 'Period'. Concatenate into a SINGLE dataframe `result`.

                ```python
                import pandas as pd
                
                clean_target = 'ENTITY_NAME_HERE'.replace(' ', '').upper()
                
                if clean_target != 'ENTITY_NAME_HERE':
                    mask_cy = (dfs[0]['Market'].str.replace(' ', '', regex=False).str.upper().str.contains(clean_target, na=False) | dfs[0]['TA_Group'].str.replace(' ', '', regex=False).str.upper().str.contains(clean_target, na=False) | dfs[0]['Resort'].str.replace(' ', '', regex=False).str.upper().str.contains(clean_target, na=False))
                    df_cy_filtered = dfs[0][mask_cy].copy()
                    
                    mask_py = (dfs[1]['Market'].str.replace(' ', '', regex=False).str.upper().str.contains(clean_target, na=False) | dfs[1]['TA_Group'].str.replace(' ', '', regex=False).str.upper().str.contains(clean_target, na=False) | dfs[1]['Resort'].str.replace(' ', '', regex=False).str.upper().str.contains(clean_target, na=False))
                    df_py_filtered = dfs[1][mask_py].copy()
                else:
                    df_cy_filtered = dfs[0].copy()
                    df_py_filtered = dfs[1].copy()
                
                df_cy_filtered['Period'] = 'CY'
                df_py_filtered['Period'] = 'PY'
                result = pd.concat([df_cy_filtered, df_py_filtered], ignore_index=True)
                ```
                """

                ai_cy_df = pd.DataFrame()
                ai_py_df = pd.DataFrame()

                try:
                    response_raw = agent.chat(hacked_prompt)
                    combined_df = extract_dataframe(response_raw)
                    
                    if isinstance(combined_df, pd.DataFrame) and 'Period' in combined_df.columns:
                        ai_cy_df = combined_df[combined_df['Period'] == 'CY']
                        ai_py_df = combined_df[combined_df['Period'] == 'PY']
                    else:
                        raise ValueError("No valid dataframe extracted.")
                        
                except Exception as e:
                    # 🌟 终极防崩溃逻辑 (Bulletproof Fallback) 🌟
                    # 如果 AI 因为闲聊而忘了写代码导致报错，系统静默拦截，直接将当前大盘数据交给后续分析！
                    ai_cy_df = df_cy_base.copy()
                    ai_py_df = df_py_base.copy()

                # --- 渲染分析与图表 ---
                if not ai_cy_df.empty or not ai_py_df.empty:
                    st.markdown("### 🌍 Executive Macro-Summary")
                    full_ui_context = f"Season: {season} | Booking Window: {start_date.strftime('%d %b %Y')} to {end_date.strftime('%d %b %Y')} | Markets: {', '.join(sel_markets) if sel_markets else 'All'}"
                    
                    insights = generate_macro_insights(ai_cy_df, ai_py_df, full_ui_context)
                    st.info(insights)
                    st.session_state.messages.append({"role": "assistant", "content": insights})
                    
                    st.markdown("### 📊 Operational Drill-down (CY)")
                    col_resort, col_ta = st.columns(2)
                    
                    with col_resort:
                        if ai_cy_df['Resort'].nunique() > 0:
                            st.plotly_chart(draw_horizontal_bar(ai_cy_df, 'Resort', 'Top Resorts by Volume (k€)', '#1D263B'), use_container_width=True)
                        
                    with col_ta:
                        if ai_cy_df['TA_Group'].nunique() > 0:
                            st.plotly_chart(draw_horizontal_bar(ai_cy_df, 'TA_Group', 'Top TA Contributors (k€)', '#A64B35'), use_container_width=True)
                else:
                    st.warning("⚠️ No data available for this analysis.")
else:
    # 🌟 迎宾大厅 UI
    welcome_html = """
    <div style="padding: 5rem 2rem; text-align: center; background: linear-gradient(135deg, #1D263B 0%, #2A3650 100%); border-radius: 16px; margin-top: 1rem; box-shadow: 0 20px 40px rgba(0,0,0,0.15);">
        <div style="font-size: 4.5rem; margin-bottom: 0.5rem; color: #A64B35; font-family: serif;">Ψ</div>
        <h1 style="font-family: 'Playfair Display', serif; font-size: 3.5rem; color: #FFFFFF; margin-bottom: 1rem; letter-spacing: 1px;">Executive Strategy Hub</h1>
        <p style="font-family: 'Inter', sans-serif; font-size: 1.15rem; color: #A4B6B0; max-width: 650px; margin: 0 auto; line-height: 1.6; font-weight: 300;">
            Elevate your revenue management. Please upload your Sales Data via the sidebar to unlock enterprise-grade pacing analytics, real-time market drill-downs, and AI-driven macroeconomic insights.
        </p>
    </div>
    """
    st.markdown(welcome_html, unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    card_style = "padding: 2rem 1.5rem; background-color: #FFFFFF; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.04); border-top: 4px solid #A64B35; height: 100%; text-align: center;"
    with c1:
        st.markdown(f'<div style="{card_style}"><div style="font-size: 2.5rem; margin-bottom: 1rem;">📅</div><h3 style="font-family: \'Playfair Display\', serif; color: #1D263B; font-size: 1.4rem; margin-bottom: 0.5rem;">Precision Pacing</h3><p style="color: #6c757d; font-size: 0.95rem; line-height: 1.5;">Align Current Year and Previous Year booking windows down to the exact day for flawless Apples-to-Apples comparisons.</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div style="{card_style}"><div style="font-size: 2.5rem; margin-bottom: 1rem;">🌍</div><h3 style="font-family: \'Playfair Display\', serif; color: #1D263B; font-size: 1.4rem; margin-bottom: 0.5rem;">Market Drill-down</h3><p style="color: #6c757d; font-size: 0.95rem; line-height: 1.5;">Instantly slice data by natural half-years (S1/S2), specific source markets, or top-performing Travel Agencies.</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div style="{card_style}"><div style="font-size: 2.5rem; margin-bottom: 1rem;">🧠</div><h3 style="font-family: \'Playfair Display\', serif; color: #1D263B; font-size: 1.4rem; margin-bottom: 0.5rem;">Macro AI Advisor</h3><p style="color: #6c757d; font-size: 0.95rem; line-height: 1.5;">Transform raw variance into boardroom-ready narratives, connecting data shifts with global macroeconomic and geopolitical trends.</p></div>', unsafe_allow_html=True)
