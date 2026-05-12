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

# --- 🌟 AI 麦肯锡级洞察生成器 (已恢复上下文全量注入) ---
def generate_pacing_insights(cy_data, py_data, context_desc):
    cy_bv = cy_data['BV'].sum() / 1000 
    py_bv = py_data['BV'].sum() / 1000
    pct = ((cy_bv - py_bv) / py_bv * 100) if py_bv > 0 else 0
    
    sys_prompt = """You are a Senior Strategy Consultant at ClubMed. Analyze the variance between Current Year (CY) and Previous Year (PY) booking pacing. 
    1. Acknowledge the specific filters active (e.g., Booking Window, Markets, Season).
    2. Detail the variance in k€ and %. 
    3. CRITICAL: Provide POSSIBLE REASONS driving this pacing shift (e.g., strategic pivot toward premium resorts, early booking campaigns, baseline anomalies).
    Write a 4-sentence boardroom-ready analysis."""
    
    user_prompt = f"UI Filters Context:\n{context_desc}\n\nCY Total BV: {cy_bv:,.0f} k€\nPY Total BV: {py_bv:,.0f} k€\nVariance: {pct:+.1f}%\n\nDetailed CY Breakdown:\n{cy_data.groupby('Dest_Type')[['BV','HN']].sum().to_string()}"
    
    try:
        resp = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)])
        return resp.content
    except:
        return f"Booking pace is currently showing a {pct:+.1f}% variance compared to the same period last year."

# --- DataFrame 提取器 ---
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

    # 🌟 统一过滤函数
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
    # 🌟 5. 智能 AI 顾问 (上下文满血复活)
    # ==========================================
    st.divider()
    st.markdown("### 🤖 Strategy Advisor")
    if "messages" not in st.session_state: st.session_state.messages = []
    
    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.markdown(m["content"])

    if prompt := st.chat_input("E.g., Analyze the variance for NJ XXY"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.write(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Analyzing targeted pacing data..."):
                
                strict_instructions = """
                You are a programmatic Python code generator. 
                YOU MUST OUTPUT EXACTLY ONE CODE BLOCK ENCLOSED IN ```python AND ```. 
                DO NOT OUTPUT ANY CONVERSATIONAL TEXT OR EXPLANATION.
                """
                
                agent = Agent([df_cy_base, df_py_base], config={"llm": llm, "save_charts": False, "custom_instructions": strict_instructions})
                
                # 告诉代码 AI：底层数据已经过滤过了，不要重复过滤！
                hacked_prompt = f"""
                User Question: {prompt}
                
                NOTE: dfs[0] (CY data) and dfs[1] (PY data) are ALREADY filtered by the UI (Season, Dates, Markets, TAs).
                
                TASK:
                1. If the user mentions a specific target (like a Market, TA_Group, or Resort), perform an uppercase fuzzy search on `Market`, `TA_Group`, and `Resort` columns to filter BOTH dfs[0] and dfs[1]. 
                2. If the user asks a general question, just use the entire dfs[0] and dfs[1].
                3. ADD a new column 'Period' to distinguish them.
                4. Concatenate into a SINGLE dataframe assigned to `result`. DO NOT RETURN A DICT.

                ```python
                import pandas as pd
                
                # 1. Decide if we need to filter further based on user question
                target = 'EXTRACTED_NAME_IF_ANY'
                
                if target and target != 'EXTRACTED_NAME_IF_ANY':
                    clean_target = target.replace(' ', '').upper()
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

                try:
                    response_raw = agent.chat(hacked_prompt)
                    combined_df = extract_dataframe(response_raw)
                    
                    if isinstance(combined_df, pd.DataFrame) and 'Period' in combined_df.columns:
                        ai_cy_df = combined_df[combined_df['Period'] == 'CY']
                        ai_py_df = combined_df[combined_df['Period'] == 'PY']
                        
                        if not ai_cy_df.empty or not ai_py_df.empty:
                            display_df = ai_cy_df.groupby(['Month','Dest_Type'])[['BV','HN']].sum().reset_index()
                            display_df['BV (k€)'] = display_df['BV'] / 1000
                            display_df = display_df.drop(columns=['BV'])
                            
                            st.markdown(f"**🔍 CY Filtered Detail:**")
                            st.dataframe(display_df.style.format({'BV (k€)': '{:,.0f}k', 'HN': '{:,.0f}'}), use_container_width=True, hide_index=True)
                            
                            st.markdown("**📉 Strategy Insight:**")
                            
                            # 🌟 恢复全量上下文给文本 AI
                            full_ui_context = f"User Question: {prompt} | Season: {season} | Booking Window: {start_date.strftime('%d %b %Y')} to {end_date.strftime('%d %b %Y')} | Markets: {', '.join(sel_markets) if sel_markets else 'All'} | TAs: {', '.join(sel_ta) if sel_ta else 'All'}"
                            
                            insights = generate_pacing_insights(ai_cy_df, ai_py_df, full_ui_context)
                            st.info(f"💡 **Executive Report:**\n\n{insights}")
                            st.session_state.messages.append({"role": "assistant", "content": insights})
                        else:
                            st.warning("⚠️ No data was found for this target in the selected timeframe.")
                    else:
                        st.markdown(str(response_raw))
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
else:
    st.info("Upload SalesData.csv to start.")
