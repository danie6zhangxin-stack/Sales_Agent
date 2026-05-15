import streamlit as st
import pandas as pd
import numpy as np 
from pandasai import Agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import plotly.graph_objects as go
import plotly.express as px
import datetime

# --- 1. Executive Visual Configuration ---
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

# --- 2. AI Engine Initialization ---
try:
    api_key = st.secrets["DEEPSEEK_API_KEY"]
except:
    api_key = "sk-xxxxxxxxxxxxxxxx" 

llm = ChatOpenAI(api_key=api_key, base_url="https://api.deepseek.com", model="deepseek-chat", temperature=0.1)

# --- 3. Core Data Cleaning (精准映射版) ---
@st.cache_data
def load_and_clean(file):
    data = pd.read_csv(file, low_memory=False)
    data.columns = [col.strip() for col in data.columns]
    
    # 🌟 1. VIP 精准映射：直接把你的真实列名绑死
    mapping = {
        'CONSUMPTION_CALENDAR[Month Name]': 'Month',
        'CONSUMPTION_CALENDAR[Consumption_month_num]': 'Month_Num',
        'CONSUMPTION_CALENDAR[Consumption_year]': 'Year',
        'CONSUMPTION_CALENDAR[Consumption_date]': 'Cons_Date',
        'SALES_CALENDAR[Sales_date]': 'Sales_Date',
        'REF_SALES_MARKET[Market]': 'Market',
        'REF_DESTINATION[Resort]': 'Resort',
        'REF_CML_AGENCY[Group_TA_cml]': 'TA_Group',
        'REF_DESTINATION[Destination type Asia]': 'Dest_Type',
        '[BVSTS___final]': 'BV_Euro',   # <--- 你的欧元专属列名
        '[HN_final]': 'HN'
    }
    data.rename(columns=mapping, inplace=True, errors='ignore')
    
    # 🌟 2. 备用模糊匹配：抓取 Locale 或者其他可能的变体
    for col in data.columns:
        if col in ['BV_Euro', 'BV_Locale', 'HN', 'Sales_Date']: continue
        c_up = col.upper()
        if 'BV' in c_up and 'EURO' in c_up and 'BV_Euro' not in data.columns: 
            data.rename(columns={col: 'BV_Euro'}, inplace=True)
        elif 'BV' in c_up and ('LOCAL' in c_up or 'LOCALE' in c_up) and 'BV_Locale' not in data.columns: 
            data.rename(columns={col: 'BV_Locale'}, inplace=True)
        elif 'HN' in c_up and 'FINAL' in c_up and 'HN' not in data.columns: 
            data.rename(columns={col: 'HN'}, inplace=True)
        elif 'SALES_DATE' in c_up and 'Sales_Date' not in data.columns: 
            data.rename(columns={col: 'Sales_Date'}, inplace=True)
            
    # 🌟 防爆底线
    if 'BV_Euro' not in data.columns: data['BV_Euro'] = 0.0
    if 'BV_Locale' not in data.columns: data['BV_Locale'] = 0.0
    if 'HN' not in data.columns: data['HN'] = 0.0
    
    for col in ['Market', 'Resort', 'TA_Group', 'Dest_Type', 'Month']:
        if col in data.columns: data[col] = data[col].astype(str).str.strip()
    
    for col in ['BV_Euro', 'BV_Locale', 'HN']:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col].astype(str).str.replace(',', '').replace(' ', ''), errors='coerce').fillna(0)
    
    data['Year'] = pd.to_numeric(data['Year'], errors='coerce').fillna(0).astype(int)
    data['Month_Num'] = pd.to_numeric(data['Month_Num'], errors='coerce').fillna(0).astype(int)
    
    if 'Sales_Date' in data.columns: data['Sales_Date'] = pd.to_datetime(data['Sales_Date'], errors='coerce')
    if 'Cons_Date' in data.columns: data['Cons_Date'] = pd.to_datetime(data['Cons_Date'], errors='coerce')
        
    return data

# --- 🌟 Plotting: Bar & Pie ---
def draw_charts(cy_df, py_df, cy_label, py_label, bv_col, dynamic_title):
    cy_g = cy_df.groupby('Dest_Type')[[bv_col]].sum().reset_index()
    py_g = py_df.groupby('Dest_Type')[[bv_col]].sum().reset_index()
    
    cy_g[bv_col] /= 1000
    py_g[bv_col] /= 1000
    
    combined = pd.merge(cy_g, py_g, on='Dest_Type', how='outer', suffixes=('_CY', '_PY')).fillna(0)
    combined['YoY_Pct'] = np.where(combined[f'{bv_col}_PY'] > 0, (combined[f'{bv_col}_CY'] - combined[f'{bv_col}_PY']) / combined[f'{bv_col}_PY'] * 100, 0)
    
    fig_bar = go.Figure()
    text_cy = [f"<b>{cy:,.0f}k<br>({pct:+.1f}%)</b>" if py > 0 else f"<b>{cy:,.0f}k</b>" for cy, py, pct in zip(combined[f'{bv_col}_CY'], combined[f'{bv_col}_PY'], combined['YoY_Pct'])]
    fig_bar.add_trace(go.Bar(x=combined['Dest_Type'], y=combined[f'{bv_col}_CY'], name=cy_label, marker_color='#1D263B', text=text_cy, textposition='auto', textfont=dict(size=14)))
    fig_bar.add_trace(go.Bar(x=combined['Dest_Type'], y=combined[f'{bv_col}_PY'], name=py_label, marker_color='#A4B6B0', text=[f"<b>{v:,.0f}k</b>" for v in combined[f'{bv_col}_PY']], textposition='auto', textfont=dict(size=14)))
    fig_bar.update_layout(title=dict(text=dynamic_title, font=dict(family="Playfair Display", size=18)), barmode='group', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', margin=dict(t=70, b=0), legend=dict(orientation="h", y=1.1, x=0.5, xanchor='center'), yaxis=dict(visible=False))

    fig_pie = px.pie(cy_g, values=bv_col, names='Dest_Type', title=f"<b>{cy_label} Share</b>", color_discrete_sequence=['#1D263B', '#A64B35', '#A4B6B0', '#EAECEF'])
    fig_pie.update_traces(textposition='inside', textinfo='percent+label', hole=.3)
    fig_pie.update_layout(showlegend=False, margin=dict(t=50, b=0, l=0, r=0))

    return fig_bar, fig_pie

# --- 🌟 AI Insights Generator ---
def generate_macro_insights(cy_data, py_data, context_desc, bv_col):
    cy_total = cy_data[bv_col].sum() / 1000
    py_total = py_data[bv_col].sum() / 1000
    pct = ((cy_total - py_total) / py_total * 100) if py_total > 0 else 0
    currency = "k€" if "Euro" in bv_col else "k (Locale)"
    
    sys_prompt = "You are a Strategy Consultant for ClubMed. Analyze the variance between CY and PY. Focus on macro-environmental shifts and strategic behavior. Use 4-5 sentences."
    user_prompt = f"Context: {context_desc}\nCurrency: {currency}\nCY: {cy_total:,.0f} | PY: {py_total:,.0f}\nVariance: {pct:+.1f}%\nBreakdown:\n{cy_data.groupby('Dest_Type')[bv_col].sum().to_string()}"
    
    try:
        resp = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)])
        return resp.content
    except:
        return "Insight generation currently unavailable."

def extract_dataframe(resp):
    if isinstance(resp, pd.DataFrame): return resp
    if isinstance(resp, dict) and 'value' in resp and isinstance(resp['value'], pd.DataFrame): return resp['value']
    try: return pd.DataFrame(resp)
    except: return None

# ==========================================
# 🌟 4. Main Business UI
# ==========================================
if uploaded_file := st.sidebar.file_uploader("Upload SalesData.csv", type=['csv']):
    df = load_and_clean(uploaded_file)
    
    with st.sidebar:
        st.markdown("### 🛠️ Global Filters")
        
        bv_selection = st.radio("Display Currency:", ["Euro (€)", "Locale (Original)"])
        bv_col = "BV_Euro" if "Euro" in bv_selection else "BV_Locale"
        currency_symbol = "€" if "Euro" in bv_selection else ""
        
        sel_markets = st.multiselect("Market Select", sorted(df['Market'].unique()))
        sel_ta = st.multiselect("Travel Agency Select", sorted(df['TA_Group'].unique()))

        st.divider()
        
        st.markdown("### 📅 Consumption Window")
        cons_mode = st.radio("Filter By:", ["Quick Select (Year/Season)", "Custom Date Range"])
        
        if cons_mode == "Quick Select (Year/Season)":
            sel_year = st.selectbox("Consumption Year", sorted(df['Year'].unique(), reverse=True), index=0)
            season = st.radio("Season Focus", ["All Year", "S1 (Jan-Jun)", "S2 (Jul-Dec)"])
            cons_start, cons_end = None, None
            cy_label, py_label = f"CY {sel_year}", f"PY {sel_year-1}"
        else:
            sel_year, season = None, None
            max_cons_date = df['Cons_Date'].max().date() if not df['Cons_Date'].dropna().empty else datetime.date.today()
            col_cs, col_ce = st.columns(2)
            cons_start = col_cs.date_input("Cons. Start", value=max_cons_date - datetime.timedelta(days=180))
            cons_end = col_ce.date_input("Cons. End", value=max_cons_date)
            cy_label, py_label = f"CY ({cons_start.year})", f"PY ({cons_start.year - 1})"

        st.divider()
        
        st.markdown("### 📅 Booking Window (Sales)")
        preset = st.selectbox("Quick Range Select", ["Last 3 Months", "Last Week", "Last 1 Month", "Custom Range"])
        max_sales_date = df['Sales_Date'].max().date() if not df['Sales_Date'].dropna().empty else datetime.date.today()
        
        if preset == "Custom Range":
            col_s, col_e = st.columns(2)
            start_date = col_s.date_input("Sales Start", value=max_sales_date - datetime.timedelta(days=90))
            end_date = col_e.date_input("Sales End", value=max_sales_date)
        else:
            days_map = {"Last Week": 7, "Last 1 Month": 30, "Last 3 Months": 90}
            start_date = max_sales_date - datetime.timedelta(days=days_map[preset])
            end_date = max_sales_date

        if start_date <= end_date:
            py_start, py_end = start_date.replace(year=start_date.year-1), end_date.replace(year=end_date.year-1)
        else: st.error("Date Error"); st.stop()

    def apply_ui_filters(input_df, c_mode, y_val, seas_val, c_start, c_end, s_start, s_end):
        d = input_df.copy()
        d = d[(d['Sales_Date'].dt.date >= s_start) & (d['Sales_Date'].dt.date <= s_end)]
        if c_mode == "Quick Select (Year/Season)":
            d = d[d['Year'] == y_val]
            if seas_val == "S1 (Jan-Jun)": d = d[d['Month_Num'].between(1, 6)]
            elif seas_val == "S2 (Jul-Dec)": d = d[d['Month_Num'].between(7, 12)]
        else:
            d = d[(d['Cons_Date'].dt.date >= c_start) & (d['Cons_Date'].dt.date <= c_end)]
            
        if sel_markets: d = d[d['Market'].isin(sel_markets)]
        if sel_ta: d = d[d['TA_Group'].isin(sel_ta)]
        return d

    df_cy_base = apply_ui_filters(df, cons_mode, sel_year, season, cons_start, cons_end, start_date, end_date)
    
    if cons_mode == "Quick Select (Year/Season)":
        py_y_val = sel_year - 1
        py_c_start, py_c_end = None, None
    else:
        py_y_val = None
        try:
            py_c_start, py_c_end = cons_start.replace(year=cons_start.year-1), cons_end.replace(year=cons_end.year-1)
        except ValueError:
            py_c_start, py_c_end = cons_start - datetime.timedelta(days=365), cons_end - datetime.timedelta(days=365)
            
    df_py_base = apply_ui_filters(df, cons_mode, py_y_val, season, py_c_start, py_c_end, py_start, py_end)

    # --- Dashboard Header & KPIs ---
    st.markdown(f"### 📈 Executive Booking Pacing: {cy_label} vs {py_label}")
    
    cy_bv, py_bv = df_cy_base[bv_col].sum() / 1000, df_py_base[bv_col].sum() / 1000
    cy_hn, py_hn = df_cy_base['HN'].sum(), df_py_base['HN'].sum()
    cy_adr, py_adr = (cy_bv * 1000) / cy_hn if cy_hn > 0 else 0, (py_bv * 1000) / py_hn if py_hn > 0 else 0

    c1, c2, c3 = st.columns(3)
    c1.metric(f"Paced BV ({bv_selection.split(' ')[0]})", f"{currency_symbol}{cy_bv:,.0f}k", f"{(cy_bv-py_bv)/py_bv*100:.1f}%" if py_bv>0 else None)
    c2.metric("Paced HN", f"{cy_hn:,.0f}", f"{(cy_hn-py_hn)/py_hn*100:.1f}%" if py_hn>0 else None)
    c3.metric("Current ADR", f"{currency_symbol}{cy_adr:,.0f}", f"{(cy_adr-py_adr)/py_adr*100:.1f}%" if py_adr>0 else None)

    # --- Charts ---
    mkt_t = ", ".join(sel_markets) if sel_markets else "All Markets"
    if cons_mode == "Quick Select (Year/Season)": cons_desc = f"{season} {sel_year}"
    else: cons_desc = f"{cons_start} to {cons_end}"
    
    chart_title = f"<b>Booking Pace by Destination Type Asia</b><br><sup style='color: gray;'>{mkt_t} | Consumption: {cons_desc}</sup>"
    
    fig_bar, fig_pie = draw_charts(df_cy_base, df_py_base, cy_label, py_label, bv_col, chart_title)
    
    col_left, col_right = st.columns([2, 1])
    with col_left: st.plotly_chart(fig_bar, use_container_width=True)
    with col_right: st.plotly_chart(fig_pie, use_container_width=True)

    # ==========================================
    # 🌟 5. AI Macro & Strategy Advisor
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
            with st.spinner("Compiling global data..."):
                strict_instr = "YOU MUST OUTPUT EXACTLY ONE CODE BLOCK ENCLOSED IN ```python AND ```. NO TEXT."
                agent = Agent([df_cy_base, df_py_base], config={"llm": llm, "save_charts": False, "custom_instructions": strict_instr})
                
                hacked_prompt = f"""User: "{prompt}". Output only code to filter dfs[0] and dfs[1] by entity name in prompt, add 'Period' column, and result = pd.concat([cy, py])."""
                
                try:
                    response_raw = agent.chat(hacked_prompt)
                    combined_df = extract_dataframe(response_raw)
                    if not isinstance(combined_df, pd.DataFrame) or 'Period' not in combined_df.columns:
                        ai_cy_df, ai_py_df = df_cy_base.copy(), df_py_base.copy()
                    else:
                        ai_cy_df = combined_df[combined_df['Period'] == 'CY']
                        ai_py_df = combined_df[combined_df['Period'] == 'PY']
                    
                    full_context = f"Question: {prompt} | Currency: {bv_selection} | Cons_Window: {cons_desc} | Sales_Window: {start_date} to {end_date}"
                    insights = generate_macro_insights(ai_cy_df, ai_py_df, full_context, bv_col)
                    st.info(insights)
                    st.session_state.messages.append({"role": "assistant", "content": insights})
                except:
                    st.error("Analysis failed. Try rephrasing.")
else:
    welcome_html = """
    <div style="padding: 5rem 2rem; text-align: center; background: linear-gradient(135deg, #1D263B 0%, #2A3650 100%); border-radius: 16px; margin-top: 1rem; box-shadow: 0 20px 40px rgba(0,0,0,0.15);">
        <div style="font-size: 4.5rem; margin-bottom: 0.5rem; color: #A64B35; font-family: serif;">Ψ</div>
        <h1 style="font-family: 'Playfair Display', serif; font-size: 3.5rem; color: #FFFFFF; margin-bottom: 1rem; letter-spacing: 1px;">Executive Intelligence Hub</h1>
        <p style="font-family: 'Inter', sans-serif; font-size: 1.15rem; color: #A4B6B0; max-width: 650px; margin: 0 auto; line-height: 1.6; font-weight: 300;">
            Elevate your sales strategy. Please upload your Sales Data via the sidebar to unlock multi-currency pacing analytics, consumption date precision, and AI-driven macroeconomic insights.
        </p>
    </div>
    """
    st.markdown(welcome_html, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    
    card_style = "padding: 2rem 1.5rem; background-color: #FFFFFF; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.04); border-top: 4px solid #A64B35; height: 100%; text-align: center;"
    
    with c1:
        st.markdown(f'''
        <div style="{card_style}">
            <div style="font-size: 2.5rem; margin-bottom: 1rem;">📅</div>
            <h3 style="font-family: 'Playfair Display', serif; color: #1D263B; font-size: 1.4rem; margin-bottom: 0.5rem;">Dual-Date Precision</h3>
            <p style="color: #6c757d; font-size: 0.95rem; line-height: 1.5;">Cross-filter by exact Booking Window and Consumption Dates to pinpoint holiday and campaign performance.</p>
        </div>
        ''', unsafe_allow_html=True)
        
    with c2:
        st.markdown(f'''
        <div style="{card_style}">
            <div style="font-size: 2.5rem; margin-bottom: 1rem;">🌍</div>
            <h3 style="font-family: 'Playfair Display', serif; color: #1D263B; font-size: 1.4rem; margin-bottom: 0.5rem;">Global Perspective</h3>
            <p style="color: #6c757d; font-size: 0.95rem; line-height: 1.5;">Instantly toggle between Euro (€) and Locale currencies, with automated visualizations for Market and category shares.</p>
        </div>
        ''', unsafe_allow_html=True)
        
    with c3:
        st.markdown(f'''
        <div style="{card_style}">
            <div style="font-size: 2.5rem; margin-bottom: 1rem;">🧠</div>
            <h3 style="font-family: 'Playfair Display', serif; color: #1D263B; font-size: 1.4rem; margin-bottom: 0.5rem;">Macro AI Advisor</h3>
            <p style="color: #6c757d; font-size: 0.95rem; line-height: 1.5;">Transform raw sales variances into boardroom-ready narratives, connecting data shifts with global macroeconomic trends.</p>
        </div>
        ''', unsafe_allow_html=True)
